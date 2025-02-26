from django.urls import path
from .views import predict_obesity

urlpatterns = [
    path('', predict_obesity, name='predict_obesity'),
]
