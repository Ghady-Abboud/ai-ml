import numpy as np
import pandas as pd

def gradient_descent(data, m_now, b_now, L = 0.01):
    m_gradient = 0
    b_gradient = 0

    n = len(data)

    for i in range(n):
        x = data.iloc[i].study_time
        y = data.iloc[i].score

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m_now = m_now - L * m_gradient
    b_now = b_now - L * b_gradient
    return m_now, b_now
