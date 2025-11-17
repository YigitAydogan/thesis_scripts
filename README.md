# thesis_scripts
This repository contains the full code used in the thesis: "BIOINSPIRED ALGORITHM TO REFLECT HUMAN NAVIGATION IN UNPREDICTABLE ENVIRONMENTS"

## Contents
- `CCA_training_final.py`: Main training script for the combined model
- `CCA_Y_eval.py`: Evaluation script for the E1 Model
- `CCA_D_eval.py`: Evaluation script for the E2 Model
- `CCA_O_eval.py`: Evaluation script for the E3 Model
- `Gaze_focus.py`: Automatic gaze location classifier script
- `ind_training.py`: Training script for the individualised models
- `ind_eval.py`: Evaluation script for the individualised models
- `trunk_training_final.py`: Training script for the trunk model
- `feet_training_final.py`: Training script for the feet model
- `DRL_final.py`: Python script that includes the Socket connection to Unity, Deep Reinforcement Learning Model training and the Foveal Vision Simulator
- `AIControl_utf8.cs`: NPC control script for the navigation simulation on the Unity application 
- `TargetManager.cs`: NPC target generation script for the navigation simulation on the Unity application
- `ScreenshotHandler_utf8.cs`: Script responsible for taking screenshots of the Unity agent and sending them to the Python DRL script through the local Socket connection  
