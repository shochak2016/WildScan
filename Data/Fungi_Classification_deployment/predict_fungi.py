"""
Fungi classifier — inference with common names and example species.

Usage:
    python predict_fungi.py photo.jpg
    python predict_fungi.py photo.jpg --level all
    python predict_fungi.py photos/*.jpg
    python predict_fungi.py photo.jpg --no-scientific

The default mode ("smart") shows class-level predictions always, and adds
order-level detail only when the model is confident (>70%). This avoids
showing unreliable fine-grained predictions.

Requires: best_model.pth, label_vocabs.json, class_mapping.json, torch, torchvision, pillow
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import convnext_tiny, convnext_small, convnext_base
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Must match train_fungi_v6_hierarchical.py exactly
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ORDER_CONFIDENCE_THRESHOLD = 0.70

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _build_convnext_base(model_name):
    if model_name == "convnext_tiny":
        return convnext_tiny(weights=None)
    if model_name == "convnext_small":
        return convnext_small(weights=None)
    if model_name == "convnext_base":
        return convnext_base(weights=None)
    raise ValueError(f"Unknown model_name: {model_name!r}")


class HierarchicalConvNeXt(nn.Module):
    """
    Exactly mirrors HierarchicalFungiNet from train_fungi_v6_hierarchical.py.

    State dict key layout (torchvision naming):
      features.*        — ConvNeXt stages
      avgpool.*         — adaptive average pool
      norm.*            — LayerNorm (classifier[0])
      dropout.*         — dropout
      head_phylum.*     — flat nn.Linear
      head_class.*      — flat nn.Linear
      head_order.*      — flat nn.Linear
    """
    def __init__(self, num_phyla, num_classes, num_orders,
                 model_name="convnext_small", dropout=0.3):
        super().__init__()
        base = _build_convnext_base(model_name)
        self.features    = base.features          # keys: features.*
        self.avgpool     = base.avgpool
        self.norm        = base.classifier[0]     # LayerNorm — keys: norm.*
        in_features      = base.classifier[2].in_features
        self.dropout     = nn.Dropout(dropout)
        self.head_phylum = nn.Linear(in_features, num_phyla)
        self.head_class  = nn.Linear(in_features, num_classes)
        self.head_order  = nn.Linear(in_features, num_orders)

    def forward(self, x):
        feats = self.features(x)
        feats = self.avgpool(feats)
        feats = self.norm(feats)
        feats = feats.flatten(1)
        feats = self.dropout(feats)
        return self.head_phylum(feats), self.head_class(feats), self.head_order(feats)


def load_model(model_path, vocabs_path, device):
    with open(vocabs_path) as f:
        vocabs = json.load(f)

    # Auto-detect backbone from class_mapping.json sitting next to the weights.
    # Falls back to convnext_small (the default used during training).
    model_name = "convnext_small"
    mapping_path = Path(model_path).parent / "class_mapping.json"
    if mapping_path.exists():
        with open(mapping_path) as f:
            mapping = json.load(f)
        arch = mapping.get("model_architecture", "")
        for candidate in ("convnext_tiny", "convnext_small", "convnext_base"):
            if candidate in arch:
                model_name = candidate
                break

    print(f"Backbone: {model_name}")
    model = HierarchicalConvNeXt(
        num_phyla=len(vocabs["phyla"]),
        num_classes=len(vocabs["classes"]),
        num_orders=len(vocabs["orders"]),
        model_name=model_name,
    )
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, vocabs


def predict(image_path, model, vocabs, device, top_k=3):
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        p_logits, c_logits, o_logits = model(tensor)

    def top_preds(logits, names, common_map, examples_map, k):
        probs = F.softmax(logits, dim=1)[0]
        topk = torch.topk(probs, min(k, len(names)))
        results = []
        for prob, idx in zip(topk.values, topk.indices):
            sci = names[idx]
            results.append({
                "common_name": common_map.get(sci, sci),
                "scientific_name": sci,
                "confidence": prob.item(),
                "examples": examples_map.get(sci, ""),
            })
        return results

    return {
        "phylum": top_preds(p_logits, vocabs["phyla"],
                            vocabs.get("phyla_common", {}),
                            vocabs.get("phyla_examples", {}), top_k),
        "class":  top_preds(c_logits, vocabs["classes"],
                            vocabs.get("classes_common", {}),
                            vocabs.get("classes_examples", {}), top_k),
        "order":  top_preds(o_logits, vocabs["orders"],
                            vocabs.get("orders_common", {}),
                            vocabs.get("orders_examples", {}), top_k),
    }


def confidence_label(conf):
    if conf >= 0.85:
        return "high"
    elif conf >= 0.60:
        return "medium"
    else:
        return "low"


def print_result(result, mode="smart", show_sci=True, top_k=3):
    """Print prediction results.

    Modes:
      smart   — always show class; show order only if confident (>70%).
                 Examples come from the most specific confident level.
      all     — show all three levels regardless of confidence.
      class   — show class only.
      order   — show order only (with confidence warning if low).
      phylum  — show phylum only.
    """

    best_class = result["class"][0]
    best_order = result["order"][0]
    best_phylum = result["phylum"][0]
    order_confident = best_order["confidence"] >= ORDER_CONFIDENCE_THRESHOLD

    if mode == "smart":
        # Always show the class-level prediction
        _print_level("Class", best_class, show_sci)

        if order_confident:
            # Order is confident — show it + use order examples (more specific)
            _print_level("Order", best_order, show_sci)
            if best_order["examples"]:
                print(f"    Could be → {best_order['examples']}")
        else:
            # Order not confident — use class examples (safer)
            if best_class["examples"]:
                print(f"    Could be → {best_class['examples']}")

        # Show additional guesses if top_k > 1
        if top_k > 1 and len(result["class"]) > 1:
            print(f"    Other possibilities:")
            for p in result["class"][1:top_k]:
                sci_part = f" ({p['scientific_name']})" if show_sci and p['common_name'] != p['scientific_name'] else ""
                print(f"      {p['common_name']}{sci_part} — {p['confidence']:.1%}")

    elif mode == "all":
        _print_level("Phylum", best_phylum, show_sci)
        if best_phylum["examples"]:
            print(f"             e.g. {best_phylum['examples']}")
        _print_level("Class", best_class, show_sci)
        if best_class["examples"]:
            print(f"             e.g. {best_class['examples']}")
        _print_level("Order", best_order, show_sci, warn_low=True)
        if best_order["examples"]:
            print(f"             e.g. {best_order['examples']}")

        if top_k > 1:
            for level_name in ("phylum", "class", "order"):
                others = result[level_name][1:top_k]
                if others:
                    print(f"    Other {level_name}:")
                    for p in others:
                        sci_part = f" ({p['scientific_name']})" if show_sci and p['common_name'] != p['scientific_name'] else ""
                        print(f"      {p['common_name']}{sci_part} — {p['confidence']:.1%}")

    else:
        # Single level mode
        level_data = result[mode]
        _print_level(mode.capitalize(), level_data[0], show_sci,
                     warn_low=(mode == "order"))
        if level_data[0]["examples"]:
            print(f"    Could be → {level_data[0]['examples']}")
        if top_k > 1:
            for p in level_data[1:top_k]:
                sci_part = f" ({p['scientific_name']})" if show_sci and p['common_name'] != p['scientific_name'] else ""
                print(f"             {p['common_name']}{sci_part} — {p['confidence']:.1%}")


def _print_level(label, pred, show_sci, warn_low=False):
    common = pred["common_name"]
    sci = pred["scientific_name"]
    conf = pred["confidence"]
    conf_tag = confidence_label(conf)

    sci_part = f" ({sci})" if show_sci and common != sci else ""
    warning = ""
    if warn_low and conf < ORDER_CONFIDENCE_THRESHOLD:
        warning = "  ⚠ low confidence"

    print(f"    {label:8s} → {common}{sci_part} — {conf:.1%} [{conf_tag}]{warning}")


def main():
    parser = argparse.ArgumentParser(description="Fungi classifier (v6)")
    parser.add_argument("images", nargs="+", help="Image file(s)")
    parser.add_argument("--model", default="best_model.pth")
    parser.add_argument("--vocabs", default="label_vocabs.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--level", choices=["smart", "phylum", "class", "order", "all"],
                        default="smart",
                        help="smart = class always + order if confident (default)")
    parser.add_argument("--no-scientific", action="store_true",
                        help="Hide scientific names")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model, vocabs = load_model(args.model, args.vocabs, device)
    print(f"Model loaded — device: {device}\n")

    show_sci = not args.no_scientific

    for img_path in args.images:
        if not Path(img_path).exists():
            print(f"  {img_path}: FILE NOT FOUND\n")
            continue

        result = predict(img_path, model, vocabs, device, top_k=args.top_k)
        print(f"  {Path(img_path).name}")
        print_result(result, mode=args.level, show_sci=show_sci, top_k=args.top_k)
        print()


if __name__ == "__main__":
    main()
