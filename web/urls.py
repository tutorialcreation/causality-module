from django.urls import path
from . import views
urlpatterns=[
    path("",views.Predict.as_view(),name='predict_form')
]