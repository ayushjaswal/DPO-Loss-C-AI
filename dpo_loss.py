import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def calculate_dpo_loss(policy_logprob_win, ref_logprob_win, 
                       policy_logprob_lose, ref_logprob_lose, beta):

    policy_chosen_ratio = policy_logprob_win - ref_logprob_win
    policy_rejected_ratio = policy_logprob_lose - ref_logprob_lose
    preference_diff = policy_chosen_ratio - policy_rejected_ratio    
    scaled_diff = beta * preference_diff
    alignment_prob = sigmoid(scaled_diff)
    
    loss = -np.log(alignment_prob)
    
    return loss


def example_usage():
    """
    Demonstrate the DPO loss calculation with example values.
    """
    print("DPO LOSS CALCULATION EXAMPLE")
    print("=" * 60)
    
    # Example scenario from Mission 8, Part 5
    beta = 0.1
    policy_logprob_chosen = -1.5
    ref_logprob_chosen = -1.2
    policy_logprob_rejected = -1.0  # Policy will prefers this 
    ref_logprob_rejected = -1.8
    
    print("\nScenario: The policy model currently PREFERS the rejected response")
    print("(Notice: policy_logprob_rejected = -1.0 is HIGHER than chosen = -1.5)")
    print("\nInputs:")
    print(f"  Beta (β):                        {beta}")
    print(f"  Policy LogProb (Chosen):         {policy_logprob_chosen}")
    print(f"  Reference LogProb (Chosen):      {ref_logprob_chosen}")
    print(f"  Policy LogProb (Rejected):       {policy_logprob_rejected}")
    print(f"  Reference LogProb (Rejected):    {ref_logprob_rejected}")
    
    # Calculate the loss
    loss = calculate_dpo_loss(
        policy_logprob_chosen,
        ref_logprob_chosen,
        policy_logprob_rejected,
        ref_logprob_rejected,
        beta
    )
    
    print(f"\nCalculated DPO Loss: {loss:.4f}")
    
    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("=" * 60)
    
    policy_chosen_ratio = policy_logprob_chosen - ref_logprob_chosen
    policy_rejected_ratio = policy_logprob_rejected - ref_logprob_rejected
    preference_diff = policy_chosen_ratio - policy_rejected_ratio
    
    print(f"\n1. Policy's relative preference for CHOSEN:   {policy_chosen_ratio:.2f}")
    print(f"2. Policy's relative preference for REJECTED: {policy_rejected_ratio:.2f}")
    print(f"3. Preference difference:                     {preference_diff:.2f}")
    
    if preference_diff < 0:
        print("\n   !!  NEGATIVE DIFFERENCE = Policy prefers rejected over chosen!")
        print("   => High loss will push model to prefer chosen response")
    else:
        print("\n   // POSITIVE DIFFERENCE = Policy prefers chosen over rejected")
        print("   => Low loss indicates good alignment")
    
    print(f"\n4. Scaled by beta (β={beta}):                 {beta * preference_diff:.2f}")
    print(f"5. After sigmoid:                             {sigmoid(beta * preference_diff):.4f}")
    print(f"6. Final loss (-log of above):               {loss:.4f}")
    
    print("\nDuring training, gradients of this loss will:")
    print("  • INCREASE probability of chosen response")
    print("  • DECREASE probability of rejected response")
    print("  • Push the model toward ethical alignment")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()