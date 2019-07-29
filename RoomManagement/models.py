from django.db import models

# Create your models here.、


# 建筑物类型
# -----------------------------------------------------------------
class BuildingType(models.Model):
    name = models.CharField(max_length=50, primary_key=True, verbose_name='建筑物类型') # 类型名称，如商场、车库

    class Meta:
        verbose_name = "建筑物类型"
        verbose_name_plural = "建筑物类型管理"

    def __str__(self):
        return self.name


# 建筑物
# -----------------------------------------------------------------
class Building(models.Model):
    name = models.CharField(max_length=50, verbose_name='建筑物名称') # 建筑物名称，如乐天百货
    avatar = models.ImageField(verbose_name=u'建筑物头像', upload_to='BuildingAvatar/%Y/%m/%d', null=True)
    address = models.CharField(max_length=100, verbose_name='详细地址') # 详细地址
    center_coordinate = models.CharField(max_length=20, verbose_name='中心点坐标') # 建筑物的中心点坐标
    outline_coordinate = models.CharField(max_length=500, verbose_name='建筑物轮廓坐标') # 建筑物轮廓坐标
    type = models.ForeignKey('BuildingType', on_delete=models.CASCADE, verbose_name='建筑物类型') # 建筑物类型

    class Meta:
        verbose_name = "建筑物"
        verbose_name_plural = "建筑物管理"

    def __str__(self):
        return self.name


# 建筑物楼层信息
# --------------------------------------------------------------------
class BuildingFloor(models.Model):
    floor = models.CharField(max_length=20,verbose_name='楼层')
    building = models.ForeignKey('Building', on_delete=models.CASCADE, verbose_name='建筑物')
    width = models.IntegerField(default=0, verbose_name='楼层的宽')
    height = models.IntegerField(default=0, verbose_name='楼层的高')
    floor_plan = models.ImageField(verbose_name=u'图片', upload_to='FloorPlan/%Y/%m/%d', null=True)

    class Meta:
        verbose_name = "建筑物楼层信息"
        verbose_name_plural = "建筑物楼层信息管理"

    def __str__(self):
        return self.floor





