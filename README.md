# QuantumCodeChallengeHackathon
Quantum Code Challenge Hackathon for Smart Cities at CTE - Cagliari Digital Lab

## Team Name

jetix

## Team Members

Gopal Ramesh Dahale

- Discord: gopal_dahale_wizard
- GitHub: Gopal-Dahale
- Email: dahalegopal27@gmail.com

## Abstract

This project explores Quantum Machine Learning (QML) for rain prediction to enhance smart city alert systems using weather data from Cagliari's urban datasets. A binary classification model was developed, labeling rainfall events based on specific weather codes. Classical Support Vector Classifier (SVC) and Quantum SVC (QSVC) models were applied, with techniques like Random Oversampling, SMOTE + Tomek, and class-weighted adjustments to handle data imbalance. Quantum boosting (QBoost) with simulated and real annealing was also implemented. Results show QSVC with class weights and QBoost with SMOTE + Tomek provided balanced recall and F1 scores, suggesting that QML could be valuable in building accurate urban alert systems for real-time applications.

## Directory structure

`SmartCityQCWizard` holds the jupyter notebooks and python files created during the challenge. Inside that directory, the new files added are:

- `weather_svc.ipynb`: Data preparation and classification with classical SVC. The notebook describes and implements various techniques to handle class imbalance. Start with this notebook to understand the flow.
- `weather_qsvc.ipynb`: Similar to above notebook except we use QSVC with Data-reuploading circuit.
- `weather_qboost_sim.ipynb`: QBoost with simulated annealing.
- `weather_qboost_dwave.ipynb`: QBoost on D-Wave's Advantage_system4.1.
- `jax_utils.py`: Helper functions for Quantum Kernel Learning.
- `qboost.py`: QBoost python classes.
- `results`: Holds the results generated during notebook execution.
- `presentation.pdf`: Presentation.

## Results

Results are sorted by recall score.

### Classical SVC

| Model                    | Recall | Precision | F1 Score | Accuracy |
|--------------------------|--------|-----------|----------|----------|
| Random Oversampling      | 1.00   | 0.45      | 0.63     | 0.98     |
| Class Weighted           | 1.00   | 0.17      | 0.29     | 0.90     |
| SMOTE + Tomek            | 0.90   | 0.29      | 0.44     | 0.95     |
| Baseline                 | 0.50   | 0.83      | 0.63     | 0.99     |

### QSVC

| Model                    | Recall | Precision | F1 Score | Accuracy |
|--------------------------|--------|-----------|----------|----------|
| Class Weighted           | 1.00   | 0.71      | 0.83     | 0.99     |
| Random Oversampling      | 0.80   | 0.47      | 0.59     | 0.98     |
| SMOTE + Tomek            | 0.80   | 0.42      | 0.55     | 0.97     |
| Baseline                 | 0.60   | 0.86      | 0.71     | 0.99     |

### QBoost (Simulated Annealing)

| Model                    | Recall | Precision | F1 Score | Accuracy |
|--------------------------|--------|-----------|----------|----------|
| SMOTE + Tomek            | 0.90   | 0.69      | 0.78     | 0.99     |
| Random Oversampling      | 0.80   | 0.67      | 0.73     | 0.99     |
| Baseline                 | 0.60   | 0.60      | 0.60     | 0.98     |

### QBoost (Advantage_system4.1)

| Model                    | Recall | Precision | F1 Score | Accuracy |
|--------------------------|--------|-----------|----------|----------|
| SMOTE + Tomek            | 1.00   | 0.56      | 0.71     | 0.98     |
| Random Oversampling      | 0.80   | 0.67      | 0.73     | 0.99     |
| Baseline                 | 0.60   | 0.60      | 0.60     | 0.98     |

## Evaluation

We note that QSVC is the strongest performer overall, achieving a balance of high recall and precision. This indicates a more reliable model for practical applications in smart city alert systems. The QBoost methods, both with simulated annealing and D-Wave's quantum annealer are also promising due a the approach to enhancing classical boosting algorithms with quantum techniques. However, they need further optimization to improve precision. While the classical SVC provides high recall score, its lower precision limits its effectiveness in real-world applications.

## Future Scope

- **Integration of Additional Features**: Using more environmental data, such as historical rainfall patterns or urban activity metrics, can enhance model performance.
- **Refining Quantum Algorithms**: Investigating optimization techniques to refine quantum model parameters. Developing better quantum algorithms suitable with large amount of data and dimensionality.

## Limitations

- **Class Imbalance**: This is natural but can be tackled with larger dataset and data preprocessing techniques.
- **Generalization**: Since the models were trained on a specific dataset, and their generalizability to other regions is uncertain.

While we demonstrate progress in using quantum algorithms for smart city applications, further refinement and exploration is needed to fully realize the model.



<hr/>

[Quantum Code Challenge](https://www.cagliaridlab.it/en/event.page?contentId=EVT881) is a entirely online hackathon on quantum computing applied to smart cities in the [CTE Cagliari DLAB](https://www.cagliaridlab.it/) project framework, organised by [CRS4](https://www.crs4.it/) in collaboration with [Open Campus](https://www.opencampus.it/) and with the endorsement of [Qitaly](https://qworld.net/qitaly/), the italian representative among the national active groups of the QWorld organization.
The event takes place entirely online from 22 to 25 October 2024!

The hackathon is bilingual, english and italian and participants from all communities are welcome.

 Objectives of the hackathon are to:
 - spread knowledge about quantum computing techniques and the type of problems it can solve, for students but also for researchers and professionals that work on other fields and maybe they can play with toy problems that are related somehow to their professional focus.
- foster the formation of a network of people who work in or are  passionate about Quantum Computing
- work on model problems and test quantum algorithms on real smart  city datasets provided by CTE-Cagliari Digital Lab, in support of the work carried out in the Quantum Technologies Lab.

Two challenges are available: Smart City QC Apprentice & Smart City QC Wizard.
- The Smart City QC Apprentice challenge involves some coding for smart city-themed toy problems, exploring universal, adiabatic and analog quantum computer paradigms, on the hashtag#Qbraid portal.
- The Smart City QC Wizard challenge is based on QML on a real dataset extracted from the sensors of the city of Cagliari. Partecipants will use the [Qbraid](https:\\www.qbraid.com) Portal and for this challenge you can cast a QML solution based on either the digital or the adiabatic paradigm.

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/crs4/QuantumCodeChallengeHackathon.git)
