"""
Step 1 (Member A): build the evaluation subset from COCO 2017 val.

Output:
  data/captions.json        [{prompt_id, image_id, caption, file_name, seed}, ...]
  data/images/<file_name>   reference images (used by member B for FID)

Usage (Colab):
  python src/prepare_data.py --n 200          # formal run
  python src/prepare_data.py --n 50           # quick pipeline test
"""
import argparse, json, os, random, urllib.request, zipfile

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/{}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    os.makedirs(f"{args.out}/images", exist_ok=True)
    ann_json = f"{args.out}/captions_val2017.json"

    # 1. download caption annotations (~240MB zip, only once)
    if not os.path.exists(ann_json):
        zpath = f"{args.out}/annotations.zip"
        print("Downloading COCO annotations ...")
        urllib.request.urlretrieve(ANN_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            with z.open("annotations/captions_val2017.json") as src, open(ann_json, "wb") as dst:
                dst.write(src.read())
        os.remove(zpath)

    coco = json.load(open(ann_json))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}

    # 2. one caption per image (first one), fixed random sample
    caps = {}
    for a in coco["annotations"]:
        caps.setdefault(a["image_id"], a["caption"].strip())
    image_ids = sorted(caps)
    random.Random(args.seed).shuffle(image_ids)
    chosen = image_ids[: args.n]

    # 3. save subset + download reference images
    records = []
    for i, img_id in enumerate(chosen):
        fname = id2file[img_id]
        path = f"{args.out}/images/{fname}"
        if not os.path.exists(path):
            urllib.request.urlretrieve(IMG_URL.format(fname), path)
        records.append({
            "prompt_id": i,
            "image_id": img_id,
            "caption": caps[img_id],
            "file_name": fname,
            "seed": 1000 + i,          # per-prompt seed, identical across samplers
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.n}")

    json.dump(records, open(f"{args.out}/captions.json", "w"), indent=2)
    print(f"Saved {len(records)} samples to {args.out}/captions.json")


if __name__ == "__main__":
    main()
