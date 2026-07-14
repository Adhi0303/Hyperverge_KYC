import cv2
import numpy as np

from src.config.settings import IMG_SIZE, IOU_THR

def dice_coeff(pred_bin, gt_bin, eps=1e-6):
    """Dice overlap of two binary arrays (or tensors): 2|A n B| / (|A| + |B|).
    1.0 = identical, 0.0 = disjoint. `eps` avoids 0/0 on empty masks."""
    pred = np.asarray(pred_bin).astype(bool)
    gt = np.asarray(gt_bin).astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    return float((2 * intersection + eps) / (pred.sum() + gt.sum() + eps))

def polys_to_instances(polygons, size=None):
    """Rasterise polygons to a LIST of per-instance boolean masks (size x size),
    one mask per polygon. Unlike polys_to_mask these are kept separate, so each
    instance can be matched one-to-one during scoring."""
    if size is None:
        size = IMG_SIZE

    instances = []
    for poly in polygons:
        if poly and len(poly) >= 3:
            mask = np.zeros((size, size), np.uint8)
            pts = (np.array(poly, np.float32) * size).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
            instances.append(mask.astype(bool))
    return instances


def mask_to_instances(binary, min_area=80):
    """Split a predicted binary union mask into separate instances via connected
    components, dropping blobs smaller than `min_area` pixels (noise)."""
    num_labels, labels = cv2.connectedComponents(binary.astype(np.uint8))

    instances = []
    for label in range(1, num_labels):                    # label 0 is the background
        component = labels == label
        if int(component.sum()) >= min_area:
            instances.append(component)
    return instances


def _iou(a, b):
    """Intersection-over-union of two boolean masks (0 if the union is empty)."""
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def match_counts(preds, gts, thr):
    """Greedy one-to-one matching between predicted and GT instances: each GT
    claims the best still-unused prediction with IoU >= thr.

    Returns (TP, FP, FN) = (matched, unmatched predictions, unmatched GTs)."""
    matched = set()
    true_positives = 0

    for gt in gts:
        best_iou = -1.0
        best_j = -1
        for j, pred in enumerate(preds):
            if j in matched:
                continue                                  # a prediction matches at most one GT
            iou = _iou(pred, gt)
            if iou >= thr and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            true_positives += 1
            matched.add(best_j)

    false_positives = len(preds) - len(matched)           # predictions that matched nothing
    false_negatives = len(gts) - true_positives           # GTs that were missed
    return true_positives, false_positives, false_negatives


def seg_prf_report(preds_list, gts_list, iou_thrs=(IOU_THR, 0.5)):
    """Aggregate instance Precision/Recall/F1 at each IoU threshold across all
    images, plus the pixel-union Dice. `preds_list` / `gts_list` are per-image
    lists of boolean instance masks."""
    report = {"images": len(gts_list), "iou": {}}

    # ---- instance-level Precision / Recall / F1, summed over all images ----
    for thr in iou_thrs:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        for preds, gts in zip(preds_list, gts_list):
            tp, fp, fn = match_counts(preds, gts, thr)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        report["iou"][thr] = {
            "P": precision, "R": recall, "F1": f1,
            "TP": total_tp, "FP": total_fp, "FN": total_fn,
        }

    # ---- pixel Dice on the union of all instances per image (instance-agnostic) ----
    dices = []
    for preds, gts in zip(preds_list, gts_list):
        if not preds and not gts:
            dices.append(1.0)                             # both empty -> perfect agreement
            continue

        pred_union = np.zeros((IMG_SIZE, IMG_SIZE), bool)
        for pred in preds:
            pred_union |= pred

        gt_union = np.zeros((IMG_SIZE, IMG_SIZE), bool)
        for gt in gts:
            gt_union |= gt

        intersection = np.logical_and(pred_union, gt_union).sum()
        dice = (2 * intersection + 1e-6) / (pred_union.sum() + gt_union.sum() + 1e-6)
        dices.append(dice)

    report["dice_mean"] = float(np.mean(dices))
    report["gt_instances"] = int(sum(len(g) for g in gts_list))
    report["pred_instances"] = int(sum(len(p) for p in preds_list))
    return report


def print_report(name, rep, primary=IOU_THR):
    """Pretty-print a seg_prf_report dict, flagging the primary IoU threshold."""
    print(f"\n=== {name}  (n={rep['images']} imgs | GT {rep['gt_instances']} / "
          f"pred {rep['pred_instances']} instances) ===")
    for thr, m in rep["iou"].items():
        star = "  <- primary" if abs(thr - primary) < 1e-9 else ""
        print(f"  IoU>={thr:.2f}   Precision {m['P']:.4f}   Recall {m['R']:.4f}   "
              f"F1 {m['F1']:.4f}   (TP {m['TP']}  FP {m['FP']}  FN {m['FN']}){star}")
    print(f"  pixel Dice (union): {rep['dice_mean']:.4f}")
