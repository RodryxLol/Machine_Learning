Examen Machine Learning – Home Credit Default Risk (Clasificación)

Proyecto completo de Machine Learning desarrollado como examen final, utilizando el dataset Home Credit Default Risk (Kaggle). El proyecto implementa un flujo end-to-end basado en la metodología CRISP-DM, estructurado como microservicios, con buenas prácticas de ingeniería de software y despliegue del modelo final como una API REST mediante Streamlit.

Contexto del Negocio:
Una entidad del sector financiero pretende optimizar el análisis de solicitudes de crédito mediante la estimación del riesgo de incumplimiento de los solicitantes. El propósito es reducir las pérdidas económicas, apoyar la automatización de las decisiones de aprobación y canalizar los casos con mayor nivel de incertidumbre hacia una evaluación manual.

Objetivo Técnico:
Desarrollar un modelo de clasificación binaria orientado a estimar la probabilidad de incumplimiento, incorporando diversas fuentes de información, aplicando técnicas de ingeniería de características a nivel agregado, abordando el desbalance entre clases y la alta dimensionalidad, y desplegando la solución a través de una API REST accesible para otros sistemas.

Supuestos y Limitaciones del Modelo:
El modelo de estimación de riesgo crediticio fue entrenado a partir de información histórica correspondiente al conjunto de datos Home Credit Default Risk, bajo el supuesto de que los patrones de comportamiento identificados permanecen relativamente estables en el tiempo. No obstante, variaciones en el entorno macroeconómico, ajustes en las políticas de otorgamiento de crédito o cambios en el perfil de los solicitantes pueden impactar negativamente su desempeño futuro.

Al tratarse de un modelo probabilístico de clasificación, los resultados obtenidos no deben considerarse determinantes por sí mismos, sino como un insumo para respaldar el proceso de toma de decisiones. La solución está concebida para apoyar el juicio humano, en particular en aquellos casos que presentan niveles de riesgo intermedios.

De manera que la incorporación de variables agregadas derivadas de historiales financieros puede generar sesgos de forma indirecta. Por ello, se recomienda implementar mecanismos de monitoreo continuo del desempeño, junto con evaluaciones y auditorías periódicas, con el objetivo de promover decisiones equitativas y responsables.

Interpretación de los clusters:

Cluster 0: personas adultas, de ingresos medios y de créditos medios. A partir 
de la tasa de morosidad puede interpretarse como un riesgo intermedio o 
moderado (6.5%).

● Cluster 1: se interpreta como un subconjunto de personas adultas pero más 
jóvenes que en el anterior, definiendo que poseen un trabajo estable y que 
solicitan créditos más grandes. De mayores ingresos que el anterior. Posee la 
tasa de morosidad más alta (10.8%), basada en la solicitud de créditos 
grandes, pero de aspectos similares al cluster anterior.

● Cluster 2: personas adultas, que solicitan créditos más grandes y de ingresos 
más altos que en los clusters anteriores. Poseen una larga estabilidad laboral. 
De tasa de morosidad más baja (5.0%), explicada por su larga estabilidad 
laboral y económica.

● Cluster 3: adultos de ingresos mayores a los de los demás clusters, solicitan 
créditos mucho más grandes que en los otros subconjuntos. Poseen empleo 
en rangos similares al cluster 0 y 1, de rango menor al anterior. Tasa de 
morosidad similar al del cluster 0 (6.7%), se interpreta como un riesgo 
intermedio a la hora de entregar préstamos.



Data Understanding (EDA)
La fase de Data Understanding se desarrolló mediante un notebook interactivo ubicado en la carpeta 01_data_understanding/Data_understanding.py

En este análisis se abordaron los siguientes puntos:

Distribución de la variable objetivo y análisis del desbalance de clases.

Identificación de valores nulos y porcentaje de missing values por variable.

Clasificación de variables numéricas y categóricas.

Análisis preliminar de correlaciones y variables relevantes.

Obtención de primeros insights para la etapa de feature engineering.

Dataset
Se utiliza el dataset Home Credit Default Risk (Kaggle), compuesto por múltiples tablas relacionales. Las principales tablas empleadas en el proyecto son:

- application_train.parquet
- application_test.parquet
- bureau.parquet
- bureau_balance.parquet
- previous_application.parquet
- POS_CASH_balance.parquet
- installments_payments.parquet
- credit_card_balance.parquet
- HomeCredit_columns_description.parquet


Todas las tablas se integran mediante feature engineering agregado previo al modelado.

Estructura del Proyecto (CRISP-DM / Microservicios)
El proyecto está organizado simulando un flujo de microservicios alineado con las fases de CRISP-DM:

EXAMEN_ML_HOME_CREDIT_FULL/

01_data_understanding/
Scripts y notebooks para análisis exploratorio de datos.

02_data_preparation/
Scripts de limpieza, integración de fuentes y feature engineering.

03_modeling/
Entrenamiento, validación y selección del modelo campeón.

04_evaluation/
Evaluación final del modelo, generación de métricas, visualizaciones y análisis de errores.

05_deployment/
Despliegue del modelo como API REST utilizando streamlit.

app.py
artifacts/
Almacenamiento de artefactos

df_scaled.csv
df_test_variables.csv
df_train_variables.csv
df_variables.csv
kmeans_model.pkl
scaler.pkl
X_scaled.csv
X_test_scaled.csv
X_train_scaled.csv

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

Deployment Despliegue del modelo seleccionado como una API REST utilizando streamlit, permitiendo su consumo externo.

Instalación
en la carpeta de visual code ejecutar comandos:
python --version (para verificar si existe python en el equipo)
python -m pip install streamlit
python -m streamlit hello (con esto se verifica que streamlit se ejecute de manera correcta como localhost)
python -m streamlit run 05_deployment/app.py (ejecuta la API)

Instalar dependencias:

pip install -r requirements.txt

Ejecución del Pipeline Completo
Paso 1 – Feature Engineering:

python 02_data_preparation/Data_preparation.py

Este paso integra todas las fuentes de datos y genera los datasets finales de entrenamiento y test en la carpeta artifacts.

Paso 2 – Entrenamiento y Selección del Modelo:

python 03_modeling/Modeling.py

Se entrenan distintos modelos, se evalúan mediante validación cruzada estratificada y se selecciona un modelo campeón.

Paso 3 – Evaluación Final:

python 04_evaluation/evaluation.py

Se generan métricas finales, curvas ROC y Precision-Recall, matriz de confusión, análisis de errores y reporte completo

Paso 4 – Despliegue como API:

python -m streamlit run 05_deployment/app.py

API disponible en:

http://localhost:8501/

Despliegue como API REST
El modelo final fue desplegado como una API REST utilizando Streamlit, permitiendo evaluar el riesgo crediticio de nuevos solicitantes mediante solicitudes HTTP.

Se debe hacer la instalación local logicamente ya que es un localhost donde se ejecuta la documentación interactiva

Se ingresan los datos del cliente, en este caso ingreso total, monto de credito, monto de anualidad, edad y años trabajados

se calcula através de una interacción de boton que aumenta o baja el valor y muestra el calculo del cluster seguido de una tabla con la información entregada

y un mensaje en textbox como por ejemplo "Clientes con ingresos altos y mayor capacidad financiera."

La documentación interactiva se encuentra disponible en:

http://localhost:8501/

Probar el endpoint POST /evaluate_risk ingresando las variables del solicitante.
