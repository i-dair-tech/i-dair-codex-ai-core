"""ri_codex_ai_core URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from auto_ml_api.data_split import data_split
from auto_ml_api.prediction import prediction
from auto_ml_api.prediction import get_columns
from auto_ml_api.statistics_generator import statistics_generator
from auto_ml_api.train import train_api,get_hyperparameters,get_models_params,save_hyperparameters,delete_user_preference,validate_hyperparameter
from auto_ml_api.get_trained_models import get_trained_models ,plot_compare,get_roc_curve
from auto_ml_api.prediction_template import get_template
from auto_ml_api.group_management import create_group,invite_member,get_groups,assign_dataset
from auto_ml_api.remove_data import remove_trained_models,remove_dataset

urlpatterns = [
    path('statistics', statistics_generator.clean_data_api, name='statistics'),
    path('dataSplit', data_split.split_data_api, name='dataSplit'),
    path('train', train_api.train, name='train'),
    path('get-hyperparameters', get_hyperparameters.get_hyperparameters, name='get-hyperparameters'),
    path('get-models-params', get_models_params.get_models_params, name='get-models-params'),
    path('save-hyperparameters', save_hyperparameters.save_hyperparameters, name='save-hyperparameters'),
    path('delete-user-preference/<int:model_id>', delete_user_preference.delete_user_preference, name='delete-user-preference'),
    path('validate-hyperparameter', validate_hyperparameter.validate_hyperparameter, name='validate-hyperparameter'),
    path("admin/", admin.site.urls),
    path('get-trained-models', get_trained_models.get_trained_models, name='getTrainedModels'),
    path('get-roc-curve', get_roc_curve.get_roc_curve, name='get-roc-curve'),
    path('plot-compare', plot_compare.plot_compare, name='plot-compare'),
    path('predict', prediction.prediction, name='prediction'),
    path('get-trained-models-by-dataset', get_trained_models.get_trained_models_by_dataset, name='getTrainedModelsByDataset'),
    path('get-columns', get_columns.get_columns, name='get-columns'),
    path('cancel-train', train_api.cancel_train, name='cancel_train'),
    path('get-group-training-in-progress', get_trained_models.get_group_training_in_progress, name='get_group_training_in_progress'),  
    path('get-template', get_template.get_template, name='get-template'), 
    path('get-training-in-progress-by-user', get_trained_models.get_training_in_progress_by_user, name='get_training_in_progress_by_user'), 
    path('create-group', create_group.create_group, name='create-group'),
    path('invite-member', invite_member.invite_member, name='invite-member'),
    path('get-groups', get_groups.get_groups, name='get-groups'),
    path('assign-dataset', assign_dataset.assign_dataset, name='assign-dataset'),
    path('remove-trained-models', remove_trained_models.remove_trained_models, name='remove-trained-models'),
    path('remove-dataset', remove_dataset.remove_dataset, name='remove-dataset'),
    
    
]
