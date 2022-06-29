from django.shortcuts import render
from django.views.generic import FormView
from .forms import PredictionForm
# Create your views here.

class Predict(FormView):
    form_class = PredictionForm
    template_name = 'predict.html'
    # success_url = reverse_lazy('<app_name>:contact-us')


    def form_valid(self, form):
        return super(PredictionForm, self).form_valid(form)
