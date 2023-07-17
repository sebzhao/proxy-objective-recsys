import numpy as np


def apk(recommendations, proxy_prefs, true_prefs):
    """
    recommendations: List[Int] = list of recommendations of items by index suggested
    proxy_prefs: List[Int] = list of items by index that the user has seen
    true_prefs: List[Int] = list of items by index that user likes
    """
    count_proxy = 0
    count_true = 0
    total_proxy = 0
    total_true = 0

    for i in range(len(recommendations)):
        seen = i + 1
        recommendation = recommendations[i]
        if recommendation in proxy_prefs:
            count_proxy += 1
            total_proxy += count_proxy / seen
        if recommendation in true_prefs:
            count_true += 1
            total_true += count_true / seen
    apk_proxy = 1 / min(len(recommendations), len(proxy_prefs)) * total_proxy if len(proxy_prefs) > 0 else 0
    apk_true = 1 / min(len(recommendations), len(true_prefs)) * total_true if len(true_prefs) > 0 else 0

    return apk_proxy, apk_true


def batch_mapk(recommendations, proxy_prefs, true_prefs):
    total_proxy, total_true = 0, 0
    for i in range(recommendations.shape[0]):
        apk_proxy, apk_true = apk(recommendations[i], proxy_prefs[i], true_prefs[i])
        total_proxy += apk_proxy
        total_true += apk_true
    return total_proxy / recommendations.shape[0], total_true / recommendations.shape[0]


def validation(watch_matrix, proxy_prefs, true_prefs, model, k=10):
    """ """
    # Get top k recommendations for each user
    item_scores = model.x @ model.y.T

    # This should be doable without 0-indexing.
    proxy_apks = []
    true_apks = []
    for i in range(watch_matrix.shape[0]):
        user_scores = item_scores[i]

        previously_scored = watch_matrix[i] == 1
        user_scores[previously_scored] = -np.inf

        # Proxy prefs
        proxy_prefs_row = proxy_prefs[i] == 1

        # Convert to item indexes
        proxy_pref_idxs = np.where(proxy_prefs_row)[0]

        # True prefs
        true_prefs_row = true_prefs[i] == 1

        # Convert to item indexes
        true_pref_idxs = np.where(true_prefs_row)[0]

        # Sort and calculate metric
        top_indexes = np.argsort(user_scores)[::-1]

        assert len(proxy_pref_idxs) >= len(true_pref_idxs)

        proxy_apk, true_apk = apk(top_indexes[:10], proxy_pref_idxs, true_pref_idxs)
        proxy_apks.append(proxy_apk)
        true_apks.append(true_apk)

    return np.mean(proxy_apks), np.mean(true_apks)


def train(model, watch_matrix, proxy_prefs, true_prefs, num_epochs=10):
    training_losses = []
    proxy_losses = []
    true_losses = []
    for i in range(num_epochs):
        print("===\nEpoch: ", i)
        model.backward()

        loss = model.loss()
        print("Training Error: ", loss)
        training_losses.append(loss)

        proxy_loss, true_loss = validation(watch_matrix, proxy_prefs, true_prefs, model)
        print("Validation Proxy MAP@K: ", proxy_loss)
        proxy_losses.append(proxy_loss)

        print("Validation True MAP@K: ", true_loss)
        true_losses.append(true_loss)

    return model, training_losses, proxy_losses, true_losses
