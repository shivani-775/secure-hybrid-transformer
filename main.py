from data_simulation import generate_data
from train import train_model
from security.attack_simulation import false_data_injection
from evaluate import evaluate
from plot_results import plot_accuracy

if __name__ == "__main__":

    # Step 1: Generate data
    generate_data()

    # Step 2: Train model
    train_model()

    # Step 3: Inject attack
    false_data_injection()

    # Step 4: Evaluate all scenarios
    normal_acc = evaluate(mode="normal")
    attack_acc = evaluate(mode="attack")
    secure_acc = evaluate(mode="secure")

    print("\n================ FINAL COMPARISON ================")
    print(f"Normal Accuracy       : {round(normal_acc*100,2)}%")
    print(f"Under Attack Accuracy : {round(attack_acc*100,2)}%")
    print(f"With AES Accuracy     : {round(secure_acc*100,2)}%")

    plot_accuracy(normal_acc, attack_acc, secure_acc)