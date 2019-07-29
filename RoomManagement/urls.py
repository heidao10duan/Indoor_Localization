"""MedicineStorageSystem URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
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

from django.conf.urls import url

from . import views

app_name='room_management'

urlpatterns = [
    url(r'^add_building/', views.add_building, name='add_building'),
    url(r'^get_building/', views.get_building, name='get_building'),
    url(r'^select_building/', views.select_building, name='select_building'),
    url(r'^delete_building/', views.delete_building, name='delete_building'),
    url(r'^query_building/', views.query_building, name='query_building'),
    url(r'^add_floor/', views.add_floor, name='add_floor'),
    url(r'^get_floor/', views.get_floor, name='get_floor'),
    url(r'^select_floor/', views.select_floor, name='select_floor'),
    url(r'^delete_floor/', views.delete_floor, name='delete_floor'),
    url(r'^add_building_type/', views.add_building_type, name='add_building_type'),
    url(r'^get_building_type/', views.get_building_type, name='get_building_type'),
    url(r'^select_building_type/', views.select_building_type, name='select_building_type'),
    url(r'^delete_building_type/', views.delete_building_type, name='delete_building_type'),

    url(r'^get_map/', views.get_map, name='get_map'),
    # 判断经纬度是否在建筑物内
    url(r'^is_in_building', views.is_in_building, name='is_in_building'),
]
