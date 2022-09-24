class DefaultClassifier:
    def get_classification(self, evaluation_dict, image_name):
        rating_keys = ['explicit', 'questionable', 'safe']
        ratings = [
            (key, evaluation_dict[f'rating:{key}']) for key in rating_keys]
        sorted_ratings = sorted(ratings, key=lambda x: x[1], reverse=True)
        print(f'{image_name}: {sorted_ratings}')
        best_rating = sorted_ratings[0]
        return best_rating[0]
