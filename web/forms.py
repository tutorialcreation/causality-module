from django import forms

class PredictionForm(forms.Form):
    texture_mean = forms.FloatField()
    area_mean = forms.FloatField()
    concavity_mean = forms.FloatField()
    area_se = forms.FloatField()
    concavity_se = forms.FloatField()
    fractal_dimension_se = forms.FloatField()
    smoothness_worst = forms.FloatField()
    concavity_worst = forms.FloatField()
    symmetry_worst = forms.FloatField()
    fractal_dimension_worst = forms.FloatField()