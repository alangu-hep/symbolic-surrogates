import numpy as np

def spearman(teacher_logits, student_logits):

    from scipy.stats import spearmanr

    rho_values = []
    
    for i in range(teacher_logits.shape):

        rho = spearmanr(student_logits[:, i], teacher_logits[:, i]).statistic
        rho_values.append(rho)
        
    return rho_values
        
def kl_div(teacher_logits, student_logits):

    from scipy.special import softmax, kl_div

    teacher_probs = softmax(teacher_logits, axis=1)
    student_probs = softmax(student_logits, axis=1)

    regular = kl_div(student_probs, teacher_probs)
    batchmean = regular.flatten().sum() / regular.shape[0]

    return batchmean