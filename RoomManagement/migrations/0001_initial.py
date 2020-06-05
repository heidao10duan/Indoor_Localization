# Generated by Django 2.1.7 on 2019-07-03 03:50

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Building',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=50, verbose_name='建筑物名称')),
                ('avatar', models.ImageField(null=True, upload_to='BuildingAvatar/%Y/%m/%d', verbose_name='建筑物头像')),
                ('address', models.CharField(max_length=100, verbose_name='详细地址')),
                ('center_coordinate', models.CharField(max_length=20, verbose_name='中心点坐标')),
                ('outline_coordinate', models.CharField(max_length=500, verbose_name='建筑物轮廓坐标')),
            ],
            options={
                'verbose_name': '建筑物',
                'verbose_name_plural': '建筑物管理',
            },
        ),
        migrations.CreateModel(
            name='BuildingFloor',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('floor', models.CharField(max_length=20, verbose_name='楼层')),
                ('width', models.IntegerField(default=0, verbose_name='楼层的宽')),
                ('height', models.IntegerField(default=0, verbose_name='楼层的高')),
                ('floor_plan', models.ImageField(null=True, upload_to='FloorPlan/%Y/%m/%d', verbose_name='图片')),
                ('building', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='RoomManagement.Building', verbose_name='建筑物')),
            ],
            options={
                'verbose_name': '建筑物楼层信息',
                'verbose_name_plural': '建筑物楼层信息管理',
            },
        ),
        migrations.CreateModel(
            name='BuildingType',
            fields=[
                ('name', models.CharField(max_length=50, primary_key=True, serialize=False, verbose_name='建筑物类型')),
            ],
            options={
                'verbose_name': '建筑物类型',
                'verbose_name_plural': '建筑物类型管理',
            },
        ),
        migrations.AddField(
            model_name='building',
            name='type',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='RoomManagement.BuildingType', verbose_name='建筑物类型'),
        ),
    ]