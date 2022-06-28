from django.db import models

# Create your models here.
class CancerMeasure(models.Model):
    texture_mean = models.FloatField()
    area_mean = models.FloatField()
    concavity_mean = models.FloatField()
    area_se = models.FloatField()
    concavity_se = models.FloatField()
    fractal_dimension_se = models.FloatField()
    smoothness_worst = models.FloatField()
    concavity_worst = models.FloatField()
    symmetry_worst = models.FloatField()
    fractal_dimension_worst = models.FloatField()