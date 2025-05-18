from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_page, name='upload_page'),
    path('solve/', views.solve_grid, name='solve_grid'),
    path('game/', views.game_page, name='game_page'),
    path('cnn-workflow/', views.cnn_workflow, name='cnn_workflow'),
]
