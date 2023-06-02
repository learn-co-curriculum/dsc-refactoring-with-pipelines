# Refactoring Your Code to Use Pipelines

## Introduction

In this lesson, you will learn how to use the core features of scikit-learn pipelines to refactor existing machine learning preprocessing code into a portable pipeline format.

## Objectives

You will be able to:

* Recall the benefits of using pipelines
* Describe the difference between a `Pipeline`, a `FeatureUnion`, and a `ColumnTransformer` in scikit-learn
* Iteratively refactor existing preprocessing code into a pipeline

## Pipelines in the Data Science Process

***If my code already works, why do I need a pipeline?***

As we covered previously, pipelines are a great way to organize your code in a DRY (don't repeat yourself) fashion. It also allows you to perform cross validation (including `GridSearchCV`) in a way that avoids leakage, because you are performing all preprocessing steps separately. Finally, it's helpful if you want to deploy your code, since it means that you only need to pickle the overall pipeline, rather than pickling the fitted model as well as all of the fitted preprocessing transformers.

***Then why not just write a pipeline from the start?***

Pipelines are designed for efficiency rather than readability, so they can become very confusing very quickly if something goes wrong. (All of the data is in NumPy arrays, not pandas dataframes, so there are no column labels by default.)

Therefore it's a good idea to write most of your code without pipelines at first, then refactor it. Eventually if you are very confident with pipelines you can save time by writing them from the start, but it's okay if you stick with the refactoring strategy!

## Code without Pipelines

Let's say we have the following (very-simple) dataset:

### Preprocessing Steps without Pipelines

These steps should be a review of preprocessing steps you have learned previously. 

#### One-Hot Encoding Categorical Data

If we just tried to apply a `StandardScaler` then a `LogisticRegression` to this dataset, we would get a `ValueError` because the values in `category` are not yet numeric.

So, let's use a `OneHotEncoder` to convert the `category` column into multiple dummy columns representing each of the categories present:

#### Feature Engineering

Let's say for the sake of example that we wanted to add a new feature called `number_odd`, which is `1` when the value of `number` is odd and `0` when the value of `number` is even. (It's not clear why this would be useful, but you can imagine a more realistic example, e.g. a boolean flag related to a purchase threshold that triggers free shipping.)

We don't want to remove `number` and replace it with `number_odd`, we want an entire new feature `number_odd` to be added.

Let's make a custom transformer for this purpose. Specifically, we'll use a `FunctionTransformer` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)). As you might have guessed from the name, a `FunctionTransformer` takes in a function as an argument (similar to the `.apply` dataframe method) and uses that function to transform the data. Unlike just using `.apply`, this transformer has the typical `.fit_transform` interface and can be used just like any other transformer (including being used in a pipeline).

#### Scaling

Then let's say we want to scale all of the features after the previous steps have been taken:

#### Bringing It All Together

Here is the full preprocessing example without a pipeline:

Now let's rewrite that with pipeline logic!

## Pieces of a Pipeline

### `Pipeline` Class

In a previous lesson, we introduced the most fundamental part of pipelines: the `Pipeline` class. This class is useful if you want to perform the same steps on every single column in your dataset. A simple example of just using a `Pipeline` would be:

```python
pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
```

However, many interesting datasets contain a mixture of kinds of data (e.g. numeric and categorical data), which means you often do not want to perform the same steps on every column. For example, one-hot encoding is useful for converting categorical data into a format that is usable in ML models, but one-hot encoding numeric data is a bad idea. You also usually want to apply different feature engineering processes to different features.

In order to apply different data cleaning and feature engineering steps to different columns, we'll use the `FeatureUnion` and `ColumnTransformer` classes.

### `ColumnTransformer` Class

The core idea of a `ColumnTransformer` is that you can **apply different preprocessing steps to different columns of the dataset**.

Looking at the preprocessing steps above, we only want to apply the `OneHotEncoder` to the `category` column, so this is a good use case for a `ColumnTransformer`:

The pipeline returns a NumPy array, but we can convert it back into a dataframe for readability if we want to:

#### Interpreting the `ColumnTransformer` Example

Let's go back and look at each of those steps more closely.

First, creating a column transformer. Here is what that code looked like above:

```python
# Create a column transformer
col_transformer = ColumnTransformer(transformers=[
    ("ohe", OneHotEncoder(categories="auto", handle_unknown="ignore"), ["category"])
], remainder="passthrough")
```

Here is the same code, spread out so we can add more comments explaining what's happening:

```python
# Create a column transformer
col_transformer = ColumnTransformer(
    # ColumnTransformer takes a list of "transformers", each of which
    # is represented by a three-tuple (not just a transformer object)
    transformers=[
        # Each tuple has three parts
        (
            # (1) This is a string representing the name. It is there
            # for readability and so you can extract information from
            # the pipeline later. scikit-learn doesn't actually care
            # what the name is.
            "ohe",
            # (2) This is the actual transformer
            OneHotEncoder(categories="auto", handle_unknown="ignore"),
            # (3) This is the list of columns that the transformer should
            # apply to. In this case, there is only one column, but it
            # still needs to be in a list
            ["category"]
        )
        # If we wanted to perform multiple different transformations
        # on different columns, we would add more tuples here
    ],
    # By default, any column that is not specified in the list of
    # transformer tuples will be dropped, but we can indicate that we
    # want them to stay as-is if we set remainder="passthrough"
    remainder="passthrough"
)
```

Next, putting the column transformer into a pipeline. Here is that original code:

```python
# Create a pipeline containing the single column transformer
pipe = Pipeline(steps=[
    ("col_transformer", col_transformer)
])
```

And again, here it is with more comments:

```python
# Create a pipeline containing the single column transformer
pipe = Pipeline(
    # Pipeline takes a list of "steps", each of which is
    # represented by a two-tuple (not just a transformer or
    # estimator object)
    steps=[
        # Each tuple has two parts
        (
            # (1) This is name of the step. Again, this is for
            # readability and retrieving steps later, so just
            # choose a name that makes sense to you
            "col_transformer",
            # (2) This is the actual transformer or estimator.
            # Note that a transformer can be a simple one like
            # StandardScaler, or a composite one like a
            # ColumnTransformer (shown here), a FeatureUnion,
            # or another Pipeline.
            # Typically the last step will be an estimator
            # (i.e. a model that makes predictions)
            col_transformer
        )
    ]
)
```

### `FeatureUnion` Class

A `FeatureUnion` **concatenates together the results of multiple different transformers**. While `Pipeline` and a `ColumnTransformer` are usually enough to perform basic *data cleaning* forms of preprocessing, it's also helpful to be able to use a `FeatureUnion` for *feature engineering* forms of preprocessing.

Let's use a `FeatureUnion` to add on the `number_odd` feature from before. Because we only want this transformation to apply to the `number` column, we need to wrap it in a `ColumnTransformer` again. Let's call this new one `feature_eng` to indicate what it is doing:

Let's also rename the other `ColumnTransformer` to `original_features_encoded` to make it clearer what it is responsible for:

Now we can combine those two into a `FeatureUnion`:

And put that `FeatureUnion` into a `Pipeline`:

Again, here it is as a more-readable dataframe:

#### Interpreting the `FeatureUnion` Example

Once more, here was the code used to create the `FeatureUnion`:

```python
feature_union = FeatureUnion(transformer_list=[
    ("encoded_features", original_features_encoded),
    ("engineered_features", feature_eng)
])
```

And here it is spread out with more comments:

```python
feature_union = FeatureUnion(
    # FeatureUnion takes a "transformer_list" containing
    # two-tuples (not just transformers)
    transformer_list=[
        # Each tuple contains two elements
        (
            # (1) Name of the feature. If you make this "drop",
            # the transformer will be ignored
            "encoded_features",
            # (2) The actual transformer (in this case, a 
            # ColumnTransformer). This one will produce the
            # numeric features as-is and the categorical
            # features one-hot encoded
            original_features_encoded
        ),
        # Here is another tuple
        (
            # (1) Name of the feature
            "engineered_features",
            # (2) The actual transformer (again, a
            # ColumnTransformer). This one will produce just
            # the flag of whether the number is even or odd
            feature_eng
        )
    ]
)
```

### Adding Final Steps to Pipeline

If we want to scale all of the features at the end, this doesn't require any additional `ColumnTransformer` or `FeatureUnion` objects, it just means we need to add another step in our `Pipeline` like this:

Additionally, if we want to add an estimator (model) as the last step, we can do it like this:

## Complete Refactored Pipeline Example

Below is the complete pipeline (without the estimator), which produces the same output as the original full preprocessing example:

Just to confirm, this produces the same result as the previous function:

We achieved the same thing in fewer lines of code, better prevention of leakage, and the ability to pickle the whole process!

Note that in both cases we returned the object or objects used for preprocessing so they could be used on test data. Without a pipeline, we would need to apply each of the transformers in `transformers`. With a pipeline, we would just need to use `pipe.transform` on test data.

## Summary

In this lesson, you learned how to make more-sophisticated pipelines using `ColumnTransformer` and `FeatureUnion` objects in addition to `Pipeline`s. We started with a preprocessing example that used sckit-learn code without pipelines, and rewrote it to use pipelines. Along the way we used `ColumnTransformer` to conditionally preprocess certain columns while leaving others alone, and `FeatureUnion` to combine engineered features with preprocessed versions of the original data. Now you should have a clearer idea of how pipelines can be used for non-trivial preprocessing tasks.
