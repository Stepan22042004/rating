from django.shortcuts import render
from ml.predict import predict


def index(request):
    prediction_label = None
    prediction_rating = None

    if request.method == 'POST':
        review = request.POST.get('input_text', '')
        prediction_label, prediction_rating = predict(review)
        if prediction_label == 1:
            prediction_label = 'positive'
        else:
            prediction_label = 'negative'

    return render(
        request,
        'index.html',
        {
            'label': prediction_label,
            'rating': prediction_rating
        }
    )
