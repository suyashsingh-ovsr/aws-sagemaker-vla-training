---
license: apache-2.0
tags:
  - robotics
  - lerobot
  - smolvla
  - bimanual
  - so-101
datasets:
  - ovsn-st-collab/blue_cylinder_grasp_v2
base_model:
  - HuggingFaceTB/SmolVLM2-500M-Video-Instruct
---

# SmolVLA — Blue Cylinder Grasp

SmolVLA fine-tuned for bimanual blue cylinder grasping on SO-101 arms.

**Base model:** `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`  
**Dataset:** `ovsn-st-collab/blue_cylinder_grasp_v2`  
**Steps:** 20,000 | **Batch size:** 8 | **LR:** 1e-4  
**Robot:** Bimanual SO-101 | **Cameras:** front, left wrist, right wrist  

## Checkpoints

| Branch | Steps |
|---|---|
| `checkpoint-005000` | 5k |
| `checkpoint-010000` | 10k |
| `checkpoint-015000` | 15k |
| `checkpoint-020000` | 20k |
