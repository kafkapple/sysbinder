import wandb
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import torch

class LoggerWrapper:
    def __init__(self, log_dir, project_name="sysbinder", experiment_name=None, config=None):
        """
        Initialize both TensorBoard and Weights & Biases loggers
        """
        # TensorBoard setup
        self.writer = SummaryWriter(log_dir)
        
        # Weights & Biases setup
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
            sync_tensorboard=True  # TensorBoard 로그를 자동으로 wandb에 동기화
        )
        
    def add_scalar(self, tag, scalar_value, global_step=None):
        """Log scalar values"""
        self.writer.add_scalar(tag, scalar_value, global_step)
        wandb.log({tag: scalar_value}, step=global_step)
    
    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        """Log multiple scalars"""
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)
        wandb.log(tag_scalar_dict, step=global_step)
    
    def add_figure(self, tag, figure, global_step=None):
        """Log matplotlib figure"""
        self.writer.add_figure(tag, figure, global_step)
        wandb.log({tag: wandb.Image(figure)}, step=global_step)
        plt.close(figure)
    
    def add_image(self, tag, img_tensor, global_step=None):
        """Log image tensor"""
        self.writer.add_image(tag, img_tensor, global_step)
        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.cpu().numpy()
        if img_tensor.shape[0] == 3:  # CHW to HWC
            img_tensor = np.transpose(img_tensor, (1, 2, 0))
        wandb.log({tag: wandb.Image(img_tensor)}, step=global_step)
    
    def add_text(self, tag, text_string, global_step=None):
        """Log text"""
        self.writer.add_text(tag, text_string, global_step)
        wandb.log({tag: text_string}, step=global_step)
    
    def add_table(self, tag, dataframe, global_step=None):
        """
        Log pandas DataFrame as a table, safely handling any circular references
        Args:
            tag: Name of the table
            dataframe: pandas DataFrame to log
            global_step: Global step value
        """
        try:
            # Convert DataFrame to markdown for better display
            markdown_table = dataframe.to_markdown(index=False)
            self.add_text(tag, markdown_table, global_step)
        except Exception as e:
            print(f"Warning: Failed to log table {tag} due to error: {str(e)}")
            try:
                # Fallback to simple string representation
                self.add_text(f"{tag}", dataframe.to_string(), global_step)
            except:
                print(f"Warning: Also failed to log table as text for {tag}")
    
    def close(self):
        """Close both loggers"""
        self.writer.close()
        wandb.finish()
