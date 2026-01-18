@echo off
echo Setting up SWARM (H-MARL Warehouse)...
python -m venv venv
call venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
jupyter nbextension enable --py widgetsnbextension --sys-prefix
echo SWARM ready! 
echo Open notebooks/01_environment.ipynb in VS Code
pause
