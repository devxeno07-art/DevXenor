import time
import requests
import joblib
import csv
import os
import sys
import json
import atexit
import subprocess
from datetime import datetime, timezone
from river import linear_model, tree, naive_bayes, metrics, preprocessing, ensemble

# ANSI escape codes for text formatting
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
WHITE = "\033[97m"
RESET = "\033[0m"

MODEL_FILES = {
    "LogisticRegression": "model_logisticregression.pkl",
    "HoeffdingTree": "model_hoeffdingtree.pkl",
    "NaiveBayes": "model_naivebayes.pkl",
    "BaggingClassifier": "model_baggingclassifier.pkl"
}

# Initialize models
models = {}
for name, fname in MODEL_FILES.items():
    if os.path.exists(fname):
        models[name] = joblib.load(fname)
    else:
        if name == "LogisticRegression":
            models[name] = preprocessing.StandardScaler() | linear_model.LogisticRegression()
        elif name == "HoeffdingTree":
            models[name] = tree.HoeffdingTreeClassifier()
        elif name == "NaiveBayes":
            models[name] = naive_bayes.GaussianNB()
        elif name == "BaggingClassifier":
            base_model = tree.HoeffdingTreeClassifier()
            models[name] = ensemble.BaggingClassifier(model=base_model, n_models=10, seed=42)

accuracies = {name: metrics.Accuracy() for name in models}
history_file = "history.csv"
pending_predictions_file = "pending_predictions.pkl"

def get_utc_time():
    """Get current UTC time"""
    return datetime.now(timezone.utc)

def get_utc_timestamp():
    """Get UTC timestamp in milliseconds with 3 second delay"""
    utc_time = get_utc_time()
    # Add 3 second delay
    timestamp = int(utc_time.timestamp() * 1000) + 3000
    return timestamp

def format_utc_time(dt=None):
    """Format UTC time for display"""
    if dt is None:
        dt = get_utc_time()
    return dt.strftime('%Y-%m-%d %H:%M:%S UTC')

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(message, color=WHITE):
    """Print a clean header"""
    print(f"{BOLD}{color}{message}{RESET}")
    print(f"{BOLD}{color}{'=' * len(message)}{RESET}")

def print_section(message, color=CYAN):
    print(f"{BOLD}{color}▶ {message}{RESET}")

def print_status(message, color=WHITE):
    print(f"{BOLD}{color}{message}{RESET}")

def print_success(message):
    print(f"{BOLD}{GREEN}✓ {message}{RESET}")

def print_warning(message):
    print(f"{BOLD}{YELLOW}⚠ {message}{RESET}")

def print_error(message):
    print(f"{BOLD}{RED}✗ {message}{RESET}")

def print_model(message, color=BLUE):
    print(f"{BOLD}{color}{message}{RESET}")

def print_result(message, is_win):
    color = GREEN if is_win else RED
    symbol = "✓" if is_win else "✗"
    print(f"{BOLD}{color}  {symbol} {message}{RESET}")

def loading_animation(message, duration=2, step=0.1):
    """Show a loading animation"""
    frames = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
    end_time = time.time() + duration
    frame_index = 0
    
    while time.time() < end_time:
        frame = frames[frame_index % len(frames)]
        print(f"\r{BOLD}{BLUE}{frame} {message}{RESET}", end="", flush=True)
        time.sleep(step)
        frame_index += 1
    
    print("\r" + " " * (len(message) + 2) + "\r", end="")  # Clear line

def fetch_data():
    """Fetch data using UTC timestamp with 3 second delay"""
    ts = get_utc_timestamp()
    url = f"https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json?ts={ts}"
    try:
        loading_animation("Fetching lottery data (UTC + 3s delay)")
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json().get('data', {}).get('list', [])
        results = []
        
        current_utc = get_utc_time()
        
        for item in data:
            try:
                number = int(item['number'])
                result = 'Big' if number >= 5 else 'Small'
                results.append({
                    'issue': item['issueNumber'],
                    'number': number,
                    'result': result,
                    'timestamp': current_utc.isoformat(),
                    'fetch_time_utc': format_utc_time(current_utc)
                })
            except (KeyError, ValueError) as e:
                print_error(f"Error processing item: {e}")
        
        print_success(f"Fetched {len(results)} entries at {format_utc_time()}")
        return results
    except Exception as e:
        print_error(f"Error fetching data: {e}")
        return []

def load_history():
    if not os.path.exists(history_file):
        return []
    try:
        loading_animation("Loading history data")
        with open(history_file, "r") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        print_error(f"Error loading history: {e}")
        return []

def save_history(data):
    try:
        fieldnames = ["issue", "number", "result", "timestamp", "fetch_time_utc"]
        with open(history_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print_success("History saved with UTC timestamps")
    except Exception as e:
        print_error(f"Error saving history: {e}")

def load_pending_predictions():
    if os.path.exists(pending_predictions_file):
        try:
            loading_animation("Loading pending predictions")
            return joblib.load(pending_predictions_file)
        except Exception as e:
            print_error(f"Error loading pending predictions: {e}")
    return {}

def save_pending_predictions(pending_predictions):
    try:
        joblib.dump(pending_predictions, pending_predictions_file)
    except Exception as e:
        print_error(f"Error saving pending predictions: {e}")

def save_accuracies():
    try:
        joblib.dump(accuracies, "accuracies.pkl")
    except Exception as e:
        print_error(f"Error saving accuracies: {e}")

def load_accuracies():
    if os.path.exists("accuracies.pkl"):
        try:
            saved_accs = joblib.load("accuracies.pkl")
            for name in models:
                if name in saved_accs:
                    if isinstance(saved_accs[name], float):
                        acc = metrics.Accuracy()
                        total = 100
                        correct = int(saved_accs[name] * total)
                        for _ in range(correct):
                            acc.update(1, 1)
                        for _ in range(total - correct):
                            acc.update(1, 0)
                        saved_accs[name] = acc
            return saved_accs
        except Exception as e:
            print_error(f"Error loading accuracies: {e}")
    return {name: metrics.Accuracy() for name in models}

def get_features(history, current_entry, is_training=False):
    features = {}
    if is_training and history:
        features['previous_number'] = int(history[-1]['number'])
    else:
        features['previous_number'] = int(current_entry['number'])
    
    if history:
        last_five = [int(h['number']) for h in history[-5:]]
        if last_five:
            features['avg_last_five'] = sum(last_five) / len(last_five)
        last_results = [h['result'] for h in history[-5:]]
        features['recent_big_count'] = last_results.count('Big')
    
    return features

def main():
    clear_screen()
    current_utc = get_utc_time()
    print_header("LOTTERY PREDICTION SYSTEM (UTC)", MAGENTA)
    print_section(f"Cycle started at {format_utc_time(current_utc)}")
    
    # Initialize JSON data structure with UTC timestamp
    json_data = {
        "last_updated_utc": current_utc.isoformat(),
        "last_updated_formatted": format_utc_time(current_utc),
        "system_timezone": "UTC",
        "delay_seconds": 3,
        "evaluation": None,
        "next_issue_prediction": None
    }
    
    # Try to load existing data
    if os.path.exists('data.json'):
        try:
            with open('data.json', 'r') as f:
                existing_data = json.load(f)
                # Preserve evaluation and prediction data
                if 'evaluation' in existing_data:
                    json_data['evaluation'] = existing_data['evaluation']
                if 'next_issue_prediction' in existing_data:
                    json_data['next_issue_prediction'] = existing_data['next_issue_prediction']
        except Exception as e:
            print_error(f"Error loading data.json: {e}")

    # Load data
    pending_predictions = load_pending_predictions()
    global accuracies
    accuracies = load_accuracies()
    
    history = load_history()
    seen_issues = {entry['issue'] for entry in history}
    new_data = fetch_data()
    
    if not new_data:
        print_warning("No data fetched from API")
        return

    new_entries = [entry for entry in reversed(new_data) if entry['issue'] not in seen_issues]
    
    if not new_entries:
        print_status("No new data to train on")
        # Still update JSON with current timestamp
        try:
            with open('data.json', 'w') as f:
                json.dump(json_data, f, indent=2)
            print_success("JSON timestamp updated")
        except Exception as e:
            print_error(f"Error saving JSON data: {e}")
        return

    print_success(f"Found {len(new_entries)} new entries")
    
    # Evaluate pending predictions and update JSON
    for entry in new_entries:
        issue = entry['issue']
        if issue in pending_predictions:
            print_section(f"Evaluating predictions for issue {issue}:", YELLOW)
            actual_result = entry['result']
            models_eval = []
            
            for model_name, pred in pending_predictions[issue].items():
                is_win = pred == actual_result
                accuracies[model_name].update(1 if actual_result == 'Big' else 0, 
                                             1 if pred == 'Big' else 0)
                
                models_eval.append({
                    "name": model_name,
                    "predicted_class": pred,
                    "actual_class": actual_result,
                    "accuracy": f"{accuracies[model_name].get():.2%}",
                    "is_correct": is_win
                })
                
                print_model(f"[{model_name}]", BLUE)
                print(f"  Predicted: {pred}, Actual: {actual_result}")
                print_result(f"Accuracy: {accuracies[model_name].get():.2%}", is_win)
            
            # Update JSON evaluation section with UTC time
            json_data["evaluation"] = {
                "issue_id": issue,
                "evaluated_at_utc": current_utc.isoformat(),
                "evaluated_at_formatted": format_utc_time(current_utc),
                "models": models_eval
            }
            
            del pending_predictions[issue]

    # Train on new entries
    if new_entries:
        print_section("Training models...", CYAN)
        for i, entry in enumerate(new_entries):
            context = history + new_entries[:i]
            if context:
                try:
                    loading_animation(f"Training on issue {entry['issue']}", duration=0.5)
                    x = get_features(context, entry, is_training=True)
                    y = 1 if entry['result'] == 'Big' else 0
                    for name, model in models.items():
                        model.learn_one(x, y)
                    print_success(f"Trained on issue {entry['issue']}")
                except Exception as e:
                    print_error(f"Error training on entry {entry['issue']}: {e}")
    
    # Update history
    history.extend(new_entries)
    save_history(history)
    
    # Make prediction for next issue with confidence
    if history:
        try:
            latest_entry = history[-1]
            latest_issue = int(latest_entry['issue'])
            next_issue = str(latest_issue + 1)
            x_next = get_features(history, latest_entry)
            
            prediction_time = get_utc_time()
            print_header(f"PREDICTION FOR NEXT ISSUE: {next_issue}", GREEN)
            print_status(f"Prediction made at: {format_utc_time(prediction_time)}")
            
            predictions = {}
            models_pred = []
            
            for name, model in models.items():
                try:
                    loading_animation(f"Predicting with {name}", duration=0.5)
                    
                    # Get prediction and confidence
                    y_pred = model.predict_one(x_next)
                    prediction = 'Big' if y_pred == 1 else 'Small'
                    predictions[name] = prediction
                    
                    # Get confidence score
                    try:
                        y_proba = model.predict_proba_one(x_next)
                        confidence = y_proba[y_pred] * 100
                        confidence_str = f"{confidence:.1f}%"
                    except (AttributeError, KeyError):
                        confidence_str = "N/A"
                    
                    models_pred.append({
                        "name": name,
                        "predicted_class": prediction,
                        "confidence": confidence_str,
                        "historical_accuracy": f"{accuracies[name].get():.2%}"
                    })
                    
                    color = BLUE if prediction == 'Big' else MAGENTA
                    print_model(f"[{name}] Prediction: {prediction} (Confidence: {confidence_str})", color)
                except Exception as e:
                    print_error(f"Error predicting with {name}: {e}")
                    predictions[name] = 'Unknown'
            
            # Update JSON prediction section with UTC time
            json_data["next_issue_prediction"] = {
                "issue_id": next_issue,
                "predicted_at_utc": prediction_time.isoformat(),
                "predicted_at_formatted": format_utc_time(prediction_time),
                "models": models_pred
            }
            
            pending_predictions[next_issue] = predictions
        except Exception as e:
            print_error(f"Error making predictions: {e}")
    
    # Save everything
    save_pending_predictions(pending_predictions)
    for name, model in models.items():
        try:
            joblib.dump(model, MODEL_FILES[name])
        except Exception as e:
            print_error(f"Error saving model {name}: {e}")
    save_accuracies()
    
    # Save JSON data with UTC timestamps
    try:
        with open('data.json', 'w') as f:
            json.dump(json_data, f, indent=2)
        print_success(f"JSON data updated with UTC time: {format_utc_time()}")
    except Exception as e:
        print_error(f"Error saving JSON data: {e}")
    
    print_success("Models, accuracies, and predictions saved")

def countdown_timer(seconds):
    """Show a countdown timer with animation and UTC time"""
    start_time = get_utc_time()
    for remaining in range(seconds, 0, -1):
        current_time = get_utc_time()
        mins, secs = divmod(remaining, 60)
        timer = f"{mins:02d}:{secs:02d}"
        print(f"\r{BOLD}{BLUE}⏱ Next cycle in: {timer} | UTC: {format_utc_time(current_time)}{RESET}", end="", flush=True)
        time.sleep(1)
    print("\r" + " " * 80 + "\r", end="")  # Clear line

def launch_background_scripts():
    """Launch cpu.py and bot.py in the background without showing logs"""
    try:
        # Suppress output using DEVNULL
        subprocess.Popen(
            [sys.executable, "cpu.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL
        )
        subprocess.Popen(
            [sys.executable, "bot.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL
        )
        print_success("Launched cpu.py and bot.py in background")
    except Exception as e:
        print_error(f"Error launching background scripts: {e}")

def terminate_background_scripts():
    """Terminate all background Python processes except current one"""
    try:
        current_pid = os.getpid()
        if os.name == 'nt':  # Windows
            os.system(f"taskkill /F /IM python.exe /FI \"PID ne {current_pid}\" > NUL 2>&1")
        else:  # Unix-like systems
            os.system(f"pkill -f 'python.*(cpu.py|bot.py)' > /dev/null 2>&1")
    except Exception as e:
        print_error(f"Error terminating background scripts: {e}")

if __name__ == "__main__":
    clear_screen()
    print_header("LOTTERY PREDICTION SYSTEM (UTC + 3s DELAY)", MAGENTA)
    print_section("Initializing...")
    print_status(f"System started at: {format_utc_time()}")
    print_status("Using UTC timezone with 3-second API delay")
    
    # Launch background scripts
    launch_background_scripts()
    
    # Ensure background scripts are terminated on exit
    atexit.register(terminate_background_scripts)
    
    try:
        while True:
            main()
            countdown_timer(10)
    except KeyboardInterrupt:
        print(f"\n{BOLD}{RED}Training stopped by user at {format_utc_time()}{RESET}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{BOLD}{RED}CRITICAL ERROR:{RESET} {e}")
        print(f"{BOLD}{YELLOW}Restarting in 15 seconds... Current UTC: {format_utc_time()}{RESET}")
        time.sleep(15)