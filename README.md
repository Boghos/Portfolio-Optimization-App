# Portfolio Optimizer

## Quick Setup Guide for Windows

Follow these steps to run the Portfolio Optimizer application:

### 1. Install Python (if not already installed)

1. Download Python 3.8 or newer from [python.org](https://www.python.org/downloads/windows/)
2. Run the installer
3. **CHECK "Add Python to PATH"** during installation
4. Click Install Now

### 2. Run the Application

1. Open Command Prompt as Administrator (right-click Command Prompt and select "Run as administrator")
2. Navigate to the project folder:
   ```
   cd path\to\portfolio-optimizer
   ```
3. Create a virtual environment:
   ```
   python -m venv venv
   ```
4. Activate the virtual environment:
   ```
   venv\Scripts\activate
   ```
5. Install all required packages:
   ```
   pip install -r requirements.txt
   ```
6. Launch the application:
   ```
   streamlit run streamlit-portfolio-optimizer.py
   ```
7. The application will open in your web browser automatically

### Troubleshooting

- If "python" is not recognized, try using "py" instead
- If you have installation errors, try updating pip first:
  ```
  python -m pip install --upgrade pip
  ```