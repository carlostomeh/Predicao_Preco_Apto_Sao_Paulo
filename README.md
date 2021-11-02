# Machine Learning Application - Predicting São Paulo Apartments Price

<span style='font-size: 2.5em'><b>São Paulo Housing 🏡</b></span><br/>
<span style='font-size: 1.5em'>Predict apartments price in São Paulo </span>

---

### **D2APR: Aprendizado de Máquina e Reconhecimento de Padrões** (IFSP, Campinas) <br/>
**Prof**: Samuel Martins (Samuka) <br/>

#### Projeto de Estudo

**Students**: Carlos Danilo Tomé e Lucas Galdino de Camargo

**Dataset**: https://www.kaggle.com/argonalyst/sao-paulo-real-estate-sale-rent-april-2019

This data is about 13,000 apartments in São Paulo City - Brazil available in Kaggle platform.

**Final Goal**: Predict housing price in São Paulo.

---

## 🎯 Files
- Sprint's : Where we develop and explore data and test algorithms.
- Final Notebook : Jupyter notebook file with all relevant information about the development process.
- My_app.py : Python application made with streamlit that running localy.

---

## 📄 Results

We test 8 regression models in this problem and the best results was made by Random Forest Regressor and the ensemble methods like AdaBoost and Gradient Bosting. Our main goal was beat baseline RSME score in test set, and we did it as you can see below: 

- Results in Test Set

| MODEL | RMSE | R² | 
| ----- | ---------- | --------- |
| Baseline - Analysts Price | 208.432 | 0.912 |
| Ada Boost Regressor | 196.459 | 0.922 |
| Random Forest Regressor | 230.837 | 0.892 |



---

## Next Steps

- Clean outliers
- Add feature selection algorithms.
- Improve streamlit application adding new geographic visualization.

