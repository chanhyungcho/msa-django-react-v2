from django.db import models

from shop.products.models import Product
from shop.s_users.models import SUser


class Cart(models.Model):
    use_in_migrations = True
    cart_id =models.AutoField(primary_key=True)

    shop_user = models.ForeignKey(SUser, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)


    class Meta:
        db_table = "shop_carts"

    def __str__(self):
        return f'{self.pk}'
