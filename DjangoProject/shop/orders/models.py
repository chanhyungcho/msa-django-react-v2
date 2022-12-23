from django.db import models

from shop.deliveries.models import Delivery
from shop.products.models import Product
from shop.s_users.models import SUser


class Order(models.Model):
    use_in_migrations = True
    order_id =models.AutoField(primary_key=True)
    created_at = models.DateTimeField(auto_now_add=True)

    shop_user = models.ForeignKey(SUser, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    delivery = models.ForeignKey(Delivery, on_delete=models.CASCADE)


    class Meta:
        db_table = "shop_orders"

    def __str__(self):
        return f'{self.pk}{self.created_at}'
