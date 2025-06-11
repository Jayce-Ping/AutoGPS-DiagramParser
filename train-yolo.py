import os
from ultralytics import YOLO
import argparse
import sys
import shutil

def train_yolo(**args):
    """
    Train a YOLO model with the specified parameters.
    """
    model_path = args['model_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    else:
        print(f"✅ Model file found at {model_path}")
    
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)

    print(f"Training model {args['name']} with the following parameters:")
    results = model.train(
        data=args['data'],
        epochs=args['epochs'],
        imgsz=args['imgsz'],
        project=args['project'],
        name=args['name'],
        device=args['device'],
    )
    return model, results

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a YOLO model.")
    parser.add_argument('--model_name', type=str, default='yolo11n-seg', help='Name of the YOLO model to train.')
    parser.add_argument('--model_path', type=str, default='models/{model_name}.pt', help='Path to the YOLO model file.')
    parser.add_argument('--device', type=int, default=-1, help='Device to use for training.')
    parser.add_argument('--dataset_dir', type=str, default='datasets/PGDP5K_yolo_seg', help='Directory of the dataset.')
    parser.add_argument('--project', type=str, default='train_{model_name}_results', help='Directory to save training results.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training.')
    parser.add_argument('--model_save_path', type=str, default='models/best-{model_name}.pt', help='Path to save the trained model.')
    
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name
    model_path = args.model_path.format(model_name=model_name)
    device = args.device
    dataset_dir = args.dataset_dir
    project = args.project.format(model_name=model_name)
    model_save_path = args.model_save_path.format(model_name=model_name)

    os.makedirs(project, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    args = {
        'model_path': model_path,
        'data': os.path.join(dataset_dir, 'pgdp_config.yaml'),
        'epochs': 300,
        'imgsz': 640,
        'project': project,
        'name': model_name,
        'device': device,
        'resume': True,  # Resume training if a previous run exists
    }
    model, results = train_yolo(**args)

    # Export the best model to ONNX format to a specific directory
    model.export(format='onnx', dynamic=True, simplify=True, save_dir=project)

    # Copy the best model to the models directory
    best_model_path = os.path.join(project, 'weights', f'best.pt')
    if os.path.exists(best_model_path):
        os.makedirs('../models', exist_ok=True)
        # Copy the best model to the target path
        shutil.copy(best_model_path, model_save_path)
        print(f"✅ Moved best model to {model_save_path}")
    else:
        print(f"⚠️ Best model not found at {best_model_path}")



if __name__ == "__main__":
    main()