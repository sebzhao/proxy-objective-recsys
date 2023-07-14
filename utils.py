import numpy as np


def apk(recommendations, true_prefs, true_prefs_rating):
    """
    recommendations: List[Int] = list of recommendations of items by index suggested
    true_prefs: List[Int] = list of ground truth items user likes
    """
    count = 0
    count_rating = 0
    total = 0
    total_rating = 0

    for i in range(len(recommendations)):
        seen = i + 1
        recommendation = recommendations[i]
        if recommendation in true_prefs:
            count += 1
            total += count / seen
        if recommendation in true_prefs_rating:
            count_rating += 1
            total_rating += count_rating / seen

    apk_proxy = 1 / min(len(recommendations), len(true_prefs)) * total if len(true_prefs) > 0 else 0
    apk_true = 1 / min(len(recommendations), len(true_prefs_rating)) * total_rating if len(true_prefs_rating) > 0 else 0

    return apk_proxy, apk_true


def validation(training_data, test_data, true_prefs, model, k=10):
    """ """
    # Get top k recommendations for each user
    item_scores = model.x @ model.y.T

    apks = []
    true_apks = []
    for user_id in range(len(item_scores)):
        user_scores = item_scores[user_id]
        previously_scored = training_data[training_data["user_id"] == user_id]["item_id"].values

        all_items = np.arange(len(user_scores))
        user_scores[np.isin(all_items, previously_scored)] = -np.inf

        # True prefs proxy
        true_prefs = test_data[test_data["user_id"] == user_id]["item_id"].values

        # True prefs rating

        # Get all items rated above 3
        true_prefs_rating = true_prefs[i, :]
        # Convert to item indexes
        true_prefs_rating = np.where(true_prefs_rating)[0]

        # Sort and calculate metric
        top_indexes = np.argsort(user_scores)[::-1]

        proxy_apk, true_apk = apk(top_indexes[:10], true_prefs, true_prefs_rating)
        apks.append(proxy_apk)
        true_apks.append(true_apk)

    return np.mean(apks), np.mean(true_apks)


def train(model, training_data, test_data, true_prefs, num_epochs=10):
    training_losses = []
    proxy_losses = []
    true_losses = []
    for i in range(num_epochs):
        print("===\nEpoch: ", i)
        model.backward()

        loss = model.loss()
        print("Training Error: ", loss)
        training_losses.append(loss)

        proxy_loss, true_loss = validation(training_data, test_data, true_prefs, model)
        print("Validation Proxy MAP@K: ", proxy_loss)
        proxy_losses.append(proxy_loss)

        print("Validation True MAP@K: ", true_loss)
        true_losses.append(true_loss)

    return model, training_losses, proxy_losses, true_losses
