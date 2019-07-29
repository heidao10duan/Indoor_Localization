from django.contrib import admin

from RoomManagement.models import Building, BuildingFloor, BuildingType

# Register your models here.


@admin.register(Building)
class BuildingAdmin(admin.ModelAdmin):
    list_display = ('name', 'address')


@admin.register(BuildingFloor)
class BuildingFloorAdmin(admin.ModelAdmin):
    list_display = ('floor', 'building')


@admin.register(BuildingType)
class BuildingTypeAdmin(admin.ModelAdmin):
    pass