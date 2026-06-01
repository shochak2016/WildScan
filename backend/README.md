# backend

API server for WildScan (TBD).

Intended to expose the trained models in [`../Models/`](../Models/) over HTTP — e.g.
an endpoint that accepts an uploaded image or audio clip, runs the relevant model
(bird sound, fungi, plant, amphibian, animal phylum, california animal), and
returns the predicted taxa + confidences consumed by the [`../frontend/`](../frontend/).

No server code yet — placeholder for the backend service.
