Examen Machine Learning – Home Credit Default Risk (Clasificación)
Proyecto completo de Machine Learning desarrollado como examen final, utilizando el dataset Home Credit Default Risk (Kaggle). El proyecto implementa un flujo end-to-end basado en la metodología CRISP-DM, estructurado como microservicios, con buenas prácticas de ingeniería de software y despliegue del modelo final como una API REST mediante FastAPI.

Contexto del Negocio
Una institución financiera busca mejorar su proceso de evaluación de solicitudes de crédito. El objetivo es estimar la probabilidad de que un solicitante incurra en incumplimiento de pago (default), con el fin de minimizar pérdidas financieras, automatizar decisiones de aprobación y derivar casos intermedios a revisión manual.

Objetivo Técnico
Construir un modelo de clasificación binaria que permita predecir el riesgo de default integrando múltiples fuentes de datos, aplicando ingeniería de características agregadas, manejando desbalance de clases y alta dimensionalidad, y sirviendo el modelo mediante una API REST consumible por otros sistemas.

Supuestos y Limitaciones del Modelo
El modelo de predicción de riesgo crediticio fue entrenado utilizando datos históricos del dataset Home Credit Default Risk, por lo que se asume que los patrones de comportamiento observados en el pasado se mantienen en el tiempo. Sin embargo, cambios en el contexto económico, políticas crediticias o comportamiento de los solicitantes podrían afectar el rendimiento futuro del modelo.

Dado que se trata de un modelo de clasificación probabilística, sus predicciones no deben ser interpretadas como decisiones definitivas, sino como una herramienta de apoyo para la toma de decisiones. El sistema está diseñado para complementar el criterio humano, especialmente en los casos clasificados como riesgo intermedio.

Asimismo, el uso de variables agregadas provenientes de historiales financieros puede introducir sesgos indirectos, por lo que se recomienda un monitoreo continuo del desempeño del modelo y auditorías periódicas para asegurar decisiones justas y responsables.

Justificación de los Umbrales de Decisión
El endpoint /evaluate_risk devuelve una probabilidad de incumplimiento junto con una decisión sugerida basada en umbrales definidos desde una perspectiva de negocio.

Los umbrales se establecieron considerando:

El desbalance de clases presente en el dataset (default como clase minoritaria).

El alto costo financiero asociado a los falsos negativos (clientes con alta probabilidad de incumplimiento clasificados como seguros).

La necesidad de incorporar una zona de revisión manual para apoyar la toma de decisiones humanas.

La lógica aplicada es la siguiente:

Probabilidad < 0.20 → APROBAR Riesgo bajo de incumplimiento.

Probabilidad entre 0.20 y 0.35 → REVISIÓN MANUAL Zona de incertidumbre donde se recomienda análisis adicional.

Probabilidad ≥ 0.35 → RECHAZAR Alto riesgo de incumplimiento.

Esta estrategia permite equilibrar automatización, control de riesgo y criterio humano.

Data Understanding (EDA)
La fase de Data Understanding se desarrolló mediante un notebook interactivo ubicado en la carpeta 01_data_understanding/01_EDA.ipynb.

En este análisis se abordaron los siguientes puntos:

Distribución de la variable objetivo (TARGET) y análisis del desbalance de clases.

Identificación de valores nulos y porcentaje de missing values por variable.

Clasificación de variables numéricas y categóricas.

Análisis preliminar de correlaciones y variables relevantes.

Obtención de primeros insights para la etapa de feature engineering.

Adicionalmente, se generó un resumen ejecutivo del análisis exploratorio en el archivo eda_summary.md.

Dataset
Se utiliza el dataset Home Credit Default Risk (Kaggle), compuesto por múltiples tablas relacionales. Las principales tablas empleadas en el proyecto son:

application_train.parquet
application_test.parquet
bureau.parquet
bureau_balance.parquet
previous_application.parquet
POS_CASH_balance.parquet
installments_payments.parquet
credit_card_balance.parquet
HomeCredit_columns_description.parquet
Todas las tablas se integran mediante feature engineering agregado previo al modelado.

Estructura del Proyecto (CRISP-DM / Microservicios)
El proyecto está organizado simulando un flujo de microservicios alineado con las fases de CRISP-DM:

EXAMEN_ML_HOME_CREDIT_FULL/

01_data_understanding/
Scripts y notebooks para análisis exploratorio de datos (EDA).

02_data_preparation/
Scripts de limpieza, integración de fuentes y feature engineering.

build_features.py
03_modeling/
Entrenamiento, validación y selección del modelo campeón.

train_and_select.py
04_evaluation/
Evaluación final del modelo, generación de métricas, visualizaciones y análisis de errores.

evaluate_champion.py
05_deployment/
Despliegue del modelo como API REST utilizando FastAPI.

app.py
artifacts/
Almacenamiento de salidas del pipeline:

train_features.parquet
test_features.parquet
champion_model.joblib
model_metrics.json
evaluation_report.json
feature_schema.json
figures/ (ROC, PR, matriz de confusión, importancia de variables)
data/
Datos en formato parquet.

docs/
Documentación adicional (opcional).

requirements.txt
README.md

Flujo CRISP-DM Implementado
El proyecto sigue el estándar CRISP-DM, organizado en una arquitectura modular que simula microservicios:

Business Understanding Definición del problema de negocio y del objetivo de predicción de riesgo crediticio.

Data Understanding Análisis exploratorio de los datos (EDA), evaluación de distribución de la variable objetivo, valores nulos y tipos de variables.

Data Preparation Limpieza, integración de múltiples fuentes de datos y creación de variables agregadas mediante feature engineering.

Modeling Entrenamiento de modelos de clasificación, manejo de desbalance y selección del modelo campeón mediante validación cruzada.

Evaluation Evaluación final del modelo con métricas, visualizaciones y análisis de errores.

Deployment Despliegue del modelo seleccionado como una API REST utilizando FastAPI, permitiendo su consumo externo.

Instalación
Crear entorno virtual (opcional):

python -m venv .venv

Activar entorno (Windows):

.venv\Scripts\Activate.ps1

Instalar dependencias:

pip install -r requirements.txt

Ejecución del Pipeline Completo
Paso 1 – Feature Engineering:

python 02_data_preparation/build_features.py

Este paso integra todas las fuentes de datos y genera los datasets finales de entrenamiento y test en la carpeta artifacts.

Paso 2 – Entrenamiento y Selección del Modelo:

python 03_modeling/train_and_select.py

Se entrenan distintos modelos, se evalúan mediante validación cruzada estratificada y se selecciona un modelo campeón. El modelo final se guarda como champion_model.joblib.

Paso 3 – Evaluación Final:

python 04_evaluation/evaluate_champion.py

Se generan métricas finales, curvas ROC y Precision-Recall, matriz de confusión, análisis de errores y reporte completo en evaluation_report.json.

Paso 4 – Despliegue como API:

python -m uvicorn 05_deployment.app:app --reload

Swagger UI disponible en:

http://127.0.0.1:8000/docs

Despliegue como API REST
El modelo final fue desplegado como una API REST utilizando FastAPI, permitiendo evaluar el riesgo crediticio de nuevos solicitantes mediante solicitudes HTTP en formato JSON.

Endpoints disponibles:

GET /health Verifica el estado de la API.

POST /evaluate_risk Evalúa el riesgo de incumplimiento de un solicitante y retorna:

probabilidad de default,

decisión sugerida de negocio,

umbrales utilizados.

Ejemplo de Request: { "features": { "AMT_CREDIT": 450000, "AMT_INCOME_TOTAL": 180000, "DAYS_BIRTH": -12000, "NAME_CONTRACT_TYPE": "Cash loans" } }

Ejemplo de Response: { "probability_default": 0.497, "decision": "RECHAZAR", "threshold_approve": 0.20, "threshold_reject": 0.35, "notes": "Regla: <0.20 APROBAR | 0.20-0.35 REVISIÓN MANUAL | >=0.35 RECHAZAR" }

La documentación interactiva se encuentra disponible en Swagger:

👉 http://127.0.0.1:8000/docs

API – Endpoints Disponibles
GET /health
Endpoint de verificación del estado de la API y carga correcta del modelo.

POST /evaluate_risk
Evalúa el riesgo crediticio de un solicitante a partir de datos enviados en formato JSON.

Reglas de negocio para la decisión:

Probabilidad < 0.20 → APROBAR
Probabilidad entre 0.20 y 0.35 → REVISIÓN MANUAL
Probabilidad ≥ 0.35 → RECHAZAR
Ejemplo de request:

{ "features": { "AMT_CREDIT": 450000, "AMT_INCOME_TOTAL": 180000, "DAYS_BIRTH": -12000, "NAME_CONTRACT_TYPE": "Cash loans" } }

Ejemplo de response:

{ "probability_default": 0.497033, "decision": "RECHAZAR", "threshold_approve": 0.2, "threshold_reject": 0.35, "notes": "Regla: <0.20 APROBAR | 0.20-0.35 REVISIÓN MANUAL | >=0.35 RECHAZAR" }

Modelado y Evaluación
Modelo principal: Logistic Regression con solver SAGA.
Manejo de desbalance: class_weight = balanced.
Codificación categórica: OneHotEncoder en formato sparse.
Validación: Stratified K-Fold Cross Validation.
Métrica principal: ROC AUC.
Análisis complementario: Precision, Recall, matriz de confusión, permutation importance y análisis de errores FP/FN.
Buenas Prácticas Implementadas
Metodología CRISP-DM completa.
Código modular y reproducible.
Separación clara por etapas del pipeline.
Manejo de alta dimensionalidad.
API documentada automáticamente con Swagger.
Reportes automáticos en artifacts.
Enfoque de negocio incorporado en la toma de decisiones.
Uso de la API
Iniciar el servidor: uvicorn 05_deployment.app:app --reload

Acceder a Swagger: http://127.0.0.1:8000/docs

Probar el endpoint POST /evaluate_risk ingresando las variables del solicitante.
