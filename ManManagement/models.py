from django.db import models

# Create your models here.


# 地图栅格及障碍物标注
# ----------------------------------------------------------------------------------
class MapGrid(models.Model):
    floor = models.ForeignKey('RoomManagement.BuildingFloor', on_delete=models.CASCADE, verbose_name='建筑物楼层')
    start_x = models.IntegerField(verbose_name='障碍物的开始行号')
    start_y = models.IntegerField(verbose_name='障碍物的结束行号')
    end_x = models.IntegerField(verbose_name='障碍物的开始列号')
    end_y = models.IntegerField(verbose_name='障碍物的结束列号')
    door_x = models.IntegerField(default=0,verbose_name='障碍物的入口行号')
    door_y = models.IntegerField(default=0,verbose_name='障碍物的入口列号')
    geo_name = models.CharField(max_length=100, null=True, verbose_name="障碍物的名称")

    class Meta:
        verbose_name = "地图栅格"
        verbose_name_plural = "地图栅格管理"

    def __str__(self):
        return str(self.id)

