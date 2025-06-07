import os
from ultralytics import YOLO
import argparse
import yaml
from pathlib import Path

def load_dataset_config(config_path):
    """
    Load dataset configuration from YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Dataset config file {config_path} does not exist.")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def evaluate_yolo(**args):
    """
    Evaluate a YOLO model with the specified parameters.
    """
    model_path = args['model_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    else:
        print(f"✅ Model file found at {model_path}")
    
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    print(f"Evaluating model on {args['data_config']} dataset...")


    model.eval()    
    # Run validation
    results = model.val(
        data=args['data_config'],
        imgsz=args['imgsz'],
        device=args['device'],
        save_json=args['save_json'],
        save_hybrid=args['save_hybrid'],
        conf=args['conf'],
        iou=args['iou'],
        max_det=args['max_det'],
        half=args['half'],
        project=args['project'],
        name=args['name'],
        exist_ok=args['exist_ok']
    )
    
    return model, results

def print_evaluation_results(results):
    """
    Print evaluation results in a formatted way.
    """
    print("\n" + "="*50)
    print("🎯 EVALUATION RESULTS")
    print("="*50)
    
    # Get metrics
    metrics = results.results_dict
    
    if 'metrics/mAP50(B)' in metrics:
        print(f"📊 Box Detection Results:")
        print(f"   mAP@0.5     : {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"   mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
    
    if 'metrics/mAP50(M)' in metrics:
        print(f"🎭 Mask Segmentation Results:")
        print(f"   mAP@0.5     : {metrics.get('metrics/mAP50(M)', 0):.4f}")
        print(f"   mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(M)', 0):.4f}")
    
    print(f"⚡ Inference Speed:")
    print(f"   Preprocess  : {results.speed.get('preprocess', 0):.2f}ms")
    print(f"   Inference   : {results.speed.get('inference', 0):.2f}ms")
    print(f"   Postprocess : {results.speed.get('postprocess', 0):.2f}ms")
    
    print("="*50)

def parse_args():
    """
    Parse command line arguments for evaluation.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model.")
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='yolo11n-seg', 
                       help='Name of the YOLO model to evaluate.')
    parser.add_argument('--model_path', type=str, default='models/best-{model_name}.pt', 
                       help='Path to the trained YOLO model file.')
    
    # Dataset parameters
    parser.add_argument('--dataset_dir', type=str, default='datasets/PGDP5K_yolo_seg', 
                       help='Directory of the dataset.')
    
    # Evaluation parameters
    parser.add_argument('--device', type=int, default=-1, 
                       help='Device to use for evaluation.')
    parser.add_argument('--imgsz', type=int, default=640, 
                       help='Image size for evaluation.')
    parser.add_argument('--conf', type=float, default=0.25, 
                       help='Confidence threshold for predictions.')
    parser.add_argument('--iou', type=float, default=0.45, 
                       help='IoU threshold for NMS.')
    parser.add_argument('--max_det', type=int, default=300, 
                       help='Maximum number of detections per image.')
    parser.add_argument('--half', action='store_true', 
                       help='Use half precision (FP16) for faster inference.')
    
    # Output parameters
    parser.add_argument('--project', type=str, default='../eval_{model_name}_results', 
                       help='Directory to save evaluation results.')
    parser.add_argument('--name', type=str, default='eval', 
                       help='Name for the evaluation run.')
    parser.add_argument('--save_json', action='store_true', 
                       help='Save results in COCO JSON format.')
    parser.add_argument('--save_hybrid', action='store_true', 
                       help='Save hybrid version of labels.')
    parser.add_argument('--exist_ok', action='store_true', 
                       help='Overwrite existing project/name.')

    return parser.parse_args()

def main():
    """
    Main evaluation function.
    """
    args = parse_args()
    
    # Format paths with model name
    model_name = args.model_name
    model_path = args.model_path.format(model_name=model_name)
    project = args.project.format(model_name=model_name)
    
    # Prepare dataset configuration path
    data_config = os.path.join(args.dataset_dir, 'pgdp_config.yaml')
    
    # Verify dataset configuration exists
    if not os.path.exists(data_config):
        print(f"❌ Dataset configuration not found at {data_config}")
        return
    
    # Load and display dataset info
    config = load_dataset_config(data_config)
    print(f"📁 Dataset: {config.get('path', 'Unknown')}")
    print(f"🏷️  Classes: {len(config.get('names', []))} classes")
    print(f"📝 Class names: {config.get('names', [])}")
    
    # Create output directory
    os.makedirs(project, exist_ok=True)
    
    # Prepare evaluation arguments
    eval_args = {
        'model_path': model_path,
        'data_config': data_config,
        'imgsz': args.imgsz,
        'device': args.device,
        'conf': args.conf,
        'iou': args.iou,
        'max_det': args.max_det,
        'half': args.half,
        'project': project,
        'name': args.name,
        'save_json': args.save_json,
        'save_hybrid': args.save_hybrid,
        'exist_ok': args.exist_ok
    }
    
    try:
        # Run evaluation
        model, results = evaluate_yolo(**eval_args)
        
        # Print results
        print_evaluation_results(results)
        
        # Save detailed results
        results_file = os.path.join(project, args.name, 'results.txt')
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            f.write(f"Model: {model_path}\n")
            f.write(f"Dataset: {data_config}\n")
            f.write(f"Image Size: {args.imgsz}\n")
            f.write(f"Confidence Threshold: {args.conf}\n")
            f.write(f"IoU Threshold: {args.iou}\n")
            f.write("\nResults:\n")
            for key, value in results.results_dict.items():
                f.write(f"{key}: {value}\n")
        
        print(f"✅ Evaluation completed! Results saved to {project}")
        print(f"📊 Detailed results saved to {results_file}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        return

if __name__ == "__main__":
    main()