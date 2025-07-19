import numpy as np
import pandas as pd
import keras
import ml_edu.experiment
import ml_edu.results
import plotly.express as px

ITERATIONS = 10
BATCH_TEST_SIZE = 10

#Here I import the dataset that I generated using other tools

chicago_taxi_dataset = pd.read_csv("housing_data_price.csv")

#and select the only columns I need (in my case every column):

training_df = chicago_taxi_dataset.loc[:, ('SIZE_SQM', 'BEDROOMS', 'PRICE_K')]

print('Read dataset completed successfully.')
print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
training_df.head(200)
print(training_df.describe(include='all'))

max_size = training_df['SIZE_SQM'].max()
print("What is the maximum size? 				Answer: ${size:.2f}".format(size = max_size))

mean_bedrooms = training_df['BEDROOMS'].mean()
print("What is the mean bedroom number? 		Answer: {bedrooms:.4f} bedrooms".format(bedrooms = mean_bedrooms))

avg_price = training_df['PRICE_K'].mean()
print("What is the average price? 		Answer: {price:.4f}k $".format(price = avg_price))

missing_values = training_df.isnull().sum().sum()
print("Are any features missing data? 				Answer:", "No" if missing_values == 0 else "Yes")

print(f"\n{training_df.corr(numeric_only = True)}")
px.scatter_matrix(training_df, dimensions=["SIZE_SQM", "BEDROOMS", "PRICE_K"]).show()

#This showed that the best dimension (argument) to use to better classify the prices is SIZE_SQM
#with the best correlation value (0.947825), and this is clear if we look at the graph

#Now we create a model that we will be teaching

def create_model(
    settings: ml_edu.experiment.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
  inputs = {name: keras.Input(shape=(1,), name=name) for name in settings.input_features}
  concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))
  outputs = keras.layers.Dense(units=1)(concatenated_inputs)
  model = keras.Model(inputs=inputs, outputs=outputs)

  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
                loss="mean_squared_error",
                metrics=metrics)

  return model

#The following function trains the model using the experiment that is provided as an argument

def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    label_name: str,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:

  features = {name: dataset[name].values for name in settings.input_features}
  label = dataset[label_name].values
  history = model.fit(x=features,
                      y=label,
                      batch_size=settings.batch_size,
                      epochs=settings.number_epochs)

  return ml_edu.experiment.Experiment(
      name=experiment_name,
      settings=settings,
      model=model,
      epochs=history.epoch,
      metrics_history=pd.DataFrame(history.history),
  )

#Here are 2 settings I will use: only using the best argument with the highest correlation, and
#the second one will have both size and number of bedrooms. The other setting will stay the same.
#Everything is inside the loop to find the best learning rate.

rmse1_10 = []
rmse2_10 = []

for i in range(1,ITERATIONS+1):
    settings_1 = ml_edu.experiment.ExperimentSettings(
        learning_rate = i/10,
        number_epochs = 20,
        batch_size = 50,
        input_features = ['SIZE_SQM']
    )

    settings_2 = ml_edu.experiment.ExperimentSettings(
        learning_rate = i/10,
        number_epochs = 20,
        batch_size = 50,
        input_features = ['SIZE_SQM', 'BEDROOMS']
    )

    metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

    #Using create_model() I create 2 different models with different setting.

    model_1 = create_model(settings_1, metrics)
    model_2 = create_model(settings_2, metrics)

    #...and conduct the experiments!
    print(f"\nExperiment number 1 - iteration {i}:")
    experiment_1 = train_model('one_feature', model_1, training_df, 'PRICE_K', settings_1)
    print("\nExperiment 1 completed successfully.")
    print(f"\nExperiment number 2 - iteration {i}:")
    experiment_2 = train_model('two_features', model_2, training_df, 'PRICE_K', settings_2)
    print("\nExperiment 2 completed successfully.")

    rmse_1 = model_1.evaluate(
        {'SIZE_SQM': training_df['SIZE_SQM'].values},
        training_df['PRICE_K'].values,
        verbose=0
    )[0]

    rmse_2 = model_2.evaluate(
        {
            'SIZE_SQM': training_df['SIZE_SQM'].values,
            'BEDROOMS': training_df['BEDROOMS'].values,
        },
        training_df['PRICE_K'].values,
        verbose=0
    )[0]

    rmse1_10.append(rmse_1)
    rmse2_10.append(rmse_2)

    #This code is used to plot the results on the graph.

    #ml_edu.results.plot_experiment_metrics(experiment_1, ['rmse'])
    #ml_edu.results.plot_model_predictions(experiment_1, training_df, 'PRICE_K')

    #ml_edu.results.plot_experiment_metrics(experiment_2, ['rmse'])
    #ml_edu.results.plot_model_predictions(experiment_2, training_df, 'PRICE_K')

#Now we take every RMSE and find the best learning rate.

min_rmse1_10 = np.min(rmse1_10)
min_rmse2_10 = np.min(rmse2_10)

best_learning_rate1 = (rmse1_10.index(min_rmse1_10)+1)/10
best_learning_rate2 = (rmse2_10.index(min_rmse2_10)+1)/10

print(f"Best learning rate for Experiment 1: {best_learning_rate1}")
print(f"Best learning rate for Experiment 2: {best_learning_rate2}")

best_settings_1 = ml_edu.experiment.ExperimentSettings(
        learning_rate = best_learning_rate1,
        number_epochs = 20,
        batch_size = 50,
        input_features = ['SIZE_SQM']
    )

best_settings_2 = ml_edu.experiment.ExperimentSettings(
    learning_rate = best_learning_rate2,
    number_epochs = 20,
    batch_size = 50,
    input_features = ['SIZE_SQM', 'BEDROOMS']
)

metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

#Creating new experiments with best learning rates...

best_model_1 = create_model(best_settings_1, metrics)
best_model_2 = create_model(best_settings_2, metrics)

#...and conduct the experiments once more!
print(f"\nBest Experiment number 1:")
experiment_1 = train_model('one_feature', best_model_1, training_df, 'PRICE_K', settings_1)
print("\nBest Experiment 1 completed successfully.")
print(f"\nBest Experiment number 2:")
experiment_2 = train_model('two_features', best_model_2, training_df, 'PRICE_K', settings_2)
print("\nBest Experiment 2 completed successfully.")

ml_edu.results.plot_experiment_metrics(experiment_1, ['rmse'])
ml_edu.results.plot_model_predictions(experiment_1, training_df, 'PRICE_K')

ml_edu.results.plot_experiment_metrics(experiment_2, ['rmse'])
ml_edu.results.plot_model_predictions(experiment_2, training_df, 'PRICE_K')

#And now, we test our results:

def format_currency(x):
  return "${:.2f}k".format(x)

def build_batch(df, batch_size):
  batch = df.sample(n=batch_size).copy()
  batch.set_index(np.arange(batch_size), inplace=True)
  return batch

def predict_price1(model, df, features, label, batch_size=BATCH_TEST_SIZE):
  batch = build_batch(df, batch_size)
  predicted_values = model.predict_on_batch(x={name: batch[name].values for name in features})

  data = {"PREDICTED_PRICE": [], "OBSERVED_PRICE": [], "L1_LOSS": [],
          features[0]: []}
  for i in range(batch_size):
    predicted = predicted_values[i][0]
    observed = batch.at[i, label]
    data["PREDICTED_PRICE"].append(format_currency(predicted))
    data["OBSERVED_PRICE"].append(format_currency(observed))
    data["L1_LOSS"].append(format_currency(abs(observed - predicted)))
    data[features[0]].append(batch.at[i, features[0]])

  output_df = pd.DataFrame(data)
  return output_df

def predict_price2(model, df, features, label, batch_size=BATCH_TEST_SIZE):
  batch = build_batch(df, batch_size)
  predicted_values = model.predict_on_batch(x={name: batch[name].values for name in features})

  data = {"PREDICTED_PRICE": [], "OBSERVED_PRICE": [], "L1_LOSS": [],
          features[0]: [], features[1]: []}
  for j in range(batch_size):
    predicted = predicted_values[j][0]
    observed = batch.at[j, label]
    data["PREDICTED_PRICE"].append(format_currency(predicted))
    data["OBSERVED_PRICE"].append(format_currency(observed))
    data["L1_LOSS"].append(format_currency(abs(observed - predicted)))
    data[features[0]].append(batch.at[j, features[0]])
    data[features[1]].append(batch.at[j, features[1]])

  output_df = pd.DataFrame(data)
  return output_df

def show_predictions(output):
  header = "-" * 62
  banner = header + "\n" + "|" + "PREDICTIONS".center(60) + "|" + "\n" + header
  print(banner)
  print(output)
  return

print("Experiment 1 Losses: ")
output1 = predict_price1(experiment_1.model, training_df, experiment_1.settings.input_features, 'PRICE_K')
show_predictions(output1)
output1['L1_LOSS'] = (
    output1['L1_LOSS']
    .str.replace('$', '', regex=False)
    .str.replace('k', '', regex=False)
    .astype(float)
)
average_l1_loss1 = output1['L1_LOSS'].mean()
print(f"\nAverage loss for experiment 1: {average_l1_loss1:.2f}k\n")

print("Experiment 2 Losses: ")
output2 = predict_price2(experiment_2.model, training_df, experiment_2.settings.input_features, 'PRICE_K')
show_predictions(output2)
output2['L1_LOSS'] = (
    output2['L1_LOSS']
    .str.replace('$', '', regex=False)
    .str.replace('k', '', regex=False)
    .astype(float)
)
average_l1_loss2 = output2['L1_LOSS'].mean()
print(f"\nAverage loss for experiment 2: {average_l1_loss2:.2f}k\n")

if average_l1_loss2 < average_l1_loss1:
    print("The second experiment is better than the first one.")
    print("You should use 2 arguments (SIZE_SQM + BEDROOMS).")
else:
    print("The first experiment is better than the second one.")
    print("You should use 1 argument SIZE_SQM, BEDROOMS only worsens the accuracy.")

def predict_custom_price(model, input_features):
    print("\n--- Predict House Price ---")
    input_data = {}

    for feature in input_features:
        while True:
            try:
                value = float(input(f"Enter value for {feature}: "))
                input_data[feature] = np.array([value])
                break
            except ValueError:
                print("Please enter a valid number.")

    prediction = model.predict(input_data)
    price = prediction[0][0]

    print(f"\nEstimated house price: {format_currency(price)}")

if average_l1_loss2 < average_l1_loss1:
    print("The second experiment is better than the first one.")
    print("You should use 2 arguments (SIZE_SQM + BEDROOMS).")
    print("")
    predict_custom_price(experiment_2.model, ['SIZE_SQM', 'BEDROOMS'])
else:
    print("The first experiment is better than the second one.")
    print("You should use 1 argument SIZE_SQM, BEDROOMS only worsens the accuracy.")
    print("")
    predict_custom_price(experiment_1.model, ['SIZE_SQM'])

#Overall, the experiment turned out to be a great practice for this basic problem. For the most part,
#the first experiment usually does a better job at predicting the prices, but sometimes it differs.
#For increasing the accuracy, we can add another argument, like, number of floors.
#This project is free to use for your purposes, so go ahead :)