"""
Master script to re-run all pruning experiments and save models.

Usage:
    python run_pruning_all.py \
        --vgg_checkpoint   path/to/vgg11_cifar10.pth \
        --resnet_checkpoint path/to/resnet18_gtsrb.pth \
        --swin_checkpoint   path/to/swin_tiny_cifar100.pth \
        --levit_checkpoint  path/to/levit384_cifar10.pth \
        --vgg_scores    vgg_weight_sensitivity_score \
        --resnet_scores resnet_weight_sensitivity_score \
        --swin_scores   swin_weight_sensitivity_score \
        --levit_scores  levit_weight_sensitivity_score \
        --gtsrb_data    path/to/GTSRB \
        --output_dir    pruned_models

Each model saves under:
    pruned_models/vgg/      (VGG-11,    CIFAR-10)
    pruned_models/resnet/   (ResNet-18, GTSRB)
    pruned_models/swin/     (Swin-Tiny, CIFAR-100)
    pruned_models/levit/    (LeViT-384, CIFAR-10)

Each strategy produces two files:
    strategy_N_<type>_fp32.pth  — pruned + fine-tuned FP32 weights + masks + metrics
    strategy_N_<type>_int8.pth  — full INT8 quantized model object

A savepoint.json per model folder tracks completed strategies so a crashed run
can be resumed by simply re-running this script with the same arguments.
"""

import argparse
import subprocess
import sys
import os


def run_script(script, extra_args, label):
    cmd = [sys.executable, script] + extra_args
    print(f"\n{'='*80}")
    print(f"  RUNNING: {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[WARNING] {label} exited with code {result.returncode}. "
              f"Fix the error, then re-run — completed strategies will be skipped.")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run all pruning experiments and save models')

    # Checkpoints
    parser.add_argument('--vgg_checkpoint',    type=str, required=True,
                        help='Path to pretrained VGG-11 checkpoint (CIFAR-10)')
    parser.add_argument('--resnet_checkpoint', type=str, required=True,
                        help='Path to pretrained ResNet-18 checkpoint (GTSRB)')
    parser.add_argument('--swin_checkpoint',   type=str, required=True,
                        help='Path to pretrained Swin-Tiny checkpoint (CIFAR-100)')
    parser.add_argument('--levit_checkpoint',  type=str, required=True,
                        help='Path to pretrained LeViT-384 checkpoint (CIFAR-10)')

    # Sensitivity score directories
    parser.add_argument('--vgg_scores',    type=str, default='vgg_weight_sensitivity_score',
                        help='Sensitivity score dir for VGG-11')
    parser.add_argument('--resnet_scores', type=str, default='resnet_weight_sensitivity_score',
                        help='Sensitivity score dir for ResNet-18')
    parser.add_argument('--swin_scores',   type=str, default='swin_weight_sensitivity_score',
                        help='Sensitivity score dir for Swin-Tiny')
    parser.add_argument('--levit_scores',  type=str, default='levit_weight_sensitivity_score',
                        help='Sensitivity score dir for LeViT-384')

    # Data
    parser.add_argument('--cifar_data',  type=str, default='./data',
                        help='Path for CIFAR-10/100 download/cache (default: ./data)')
    parser.add_argument('--gtsrb_data',  type=str, required=True,
                        help='Path to GTSRB dataset root directory')

    # Common settings
    parser.add_argument('--output_dir', type=str, default='pruned_models',
                        help='Root directory for saved models (default: pruned_models)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs',     type=int, default=30,
                        help='Fine-tuning epochs per strategy (default: 30)')
    parser.add_argument('--device',     type=str, default='cuda',
                        help='Device: cuda or cpu (default: cuda)')

    # Which models to run
    parser.add_argument('--skip_vgg',    action='store_true', help='Skip VGG-11')
    parser.add_argument('--skip_resnet', action='store_true', help='Skip ResNet-18')
    parser.add_argument('--skip_swin',   action='store_true', help='Skip Swin-Tiny')
    parser.add_argument('--skip_levit',  action='store_true', help='Skip LeViT-384')

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    statuses = {}

    # ── VGG-11 ────────────────────────────────────────────────────────────────
    if not args.skip_vgg:
        statuses['vgg'] = run_script(
            os.path.join(script_dir, 'VGG_multi_Strategy.py'),
            [
                '--checkpoint', args.vgg_checkpoint,
                '--score_dir',  args.vgg_scores,
                '--data_path',  args.cifar_data,
                '--batch_size', str(args.batch_size),
                '--epochs',     str(args.epochs),
                '--device',     args.device,
                '--output_dir', os.path.join(args.output_dir, 'vgg'),
            ],
            'VGG-11 (CIFAR-10)'
        )

    # ── ResNet-18 ─────────────────────────────────────────────────────────────
    if not args.skip_resnet:
        statuses['resnet'] = run_script(
            os.path.join(script_dir, 'resnet_multi_strategy.py'),
            [
                '--checkpoint', args.resnet_checkpoint,
                '--score_dir',  args.resnet_scores,
                '--data_path',  args.gtsrb_data,
                '--batch_size', str(args.batch_size),
                '--epochs',     str(args.epochs),
                '--device',     args.device,
                '--output_dir', os.path.join(args.output_dir, 'resnet'),
            ],
            'ResNet-18 (GTSRB)'
        )

    # ── Swin-Tiny ─────────────────────────────────────────────────────────────
    if not args.skip_swin:
        statuses['swin'] = run_script(
            os.path.join(script_dir, 'swin_multi_strategy.py'),
            [
                '--checkpoint', args.swin_checkpoint,
                '--score_dir',  args.swin_scores,
                '--data_path',  args.cifar_data,
                '--batch_size', str(args.batch_size),
                '--epochs',     str(args.epochs),
                '--device',     args.device,
                '--output_dir', os.path.join(args.output_dir, 'swin'),
            ],
            'Swin-Tiny (CIFAR-100)'
        )

    # ── LeViT-384 ─────────────────────────────────────────────────────────────
    if not args.skip_levit:
        statuses['levit'] = run_script(
            os.path.join(script_dir, 'levit_multi_strategy.py'),
            [
                '--checkpoint', args.levit_checkpoint,
                '--score_dir',  args.levit_scores,
                '--data_path',  args.cifar_data,
                '--batch_size', str(args.batch_size),
                '--epochs',     str(args.epochs),
                '--device',     args.device,
                '--output_dir', os.path.join(args.output_dir, 'levit'),
            ],
            'LeViT-384 (CIFAR-10)'
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for model, code in statuses.items():
        status = 'OK' if code == 0 else f'FAILED (exit {code})'
        out_path = os.path.join(args.output_dir, model)
        print(f"  {model:<10} {status:<20} → {out_path}")
    print()
    print("Saved files per strategy (inside each model folder):")
    print("  strategy_N_<type>_fp32.pth  — FP32 weights + masks + accuracy metrics")
    print("  strategy_N_<type>_int8.pth  — full INT8 quantized model")
    print("  savepoint.json              — tracks completed strategies for resume")
    print()

    # How to load
    print("To load a saved model:")
    print("  # FP32 pruned model:")
    print("  import torch, timm")
    print("  ckpt = torch.load('pruned_models/vgg/strategy_1_max_aggressive_fp32.pth')")
    print("  model = timm.create_model('vgg11_bn', pretrained=False, num_classes=10)")
    print("  model.load_state_dict(ckpt['state_dict'])")
    print("  print('FP32 accuracy:', ckpt['fp32_accuracy'], '%')")
    print()
    print("  # INT8 quantized model (full model object, not just weights):")
    print("  model_int8 = torch.load('pruned_models/vgg/strategy_1_max_aggressive_int8.pth')")
    print("  model_int8.eval()")


if __name__ == '__main__':
    main()
