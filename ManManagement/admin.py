from django.contrib import admin

from ManManagement.models import MapGrid
# Register your models here.


@admin.register(MapGrid)
class MapGridAdmin(admin.ModelAdmin):
    list_display = ('id', 'floor', 'geo_name', 'start_x', 'start_y', 'end_x', 'end_y')