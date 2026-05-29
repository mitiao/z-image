"""Z-Image API client."""

import argparse
from pathlib import Path
import urllib3
urllib3.disable_warnings()

import requests


def main():
    parser = argparse.ArgumentParser(description="Generate image via Z-Image API")
    parser.add_argument("prompt", type=str, help="Text prompt")
    parser.add_argument("--steps", type=int, default=8, help="Inference steps (default: 8)")
    parser.add_argument("--height", type=int, default=1024, help="Image height (default: 1024)")
    parser.add_argument("--width", type=int, default=1024, help="Image width (default: 1024)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--guidance", type=float, default=0.0, help="Guidance scale (default: 0.0)")
    parser.add_argument("--output", "-o", type=str, default="output.png", help="Output path (default: output.png)")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()

    resp = requests.post(
        f"{args.url}/v1/images/generations",
        json={
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance,
            "seed": args.seed,
        },
        verify=False,
        timeout=600
    )
    resp.raise_for_status()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    print(f"Saved to {out_path} ({len(resp.content) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
