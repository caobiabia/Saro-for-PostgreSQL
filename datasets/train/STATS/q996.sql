select  count(*) from badges as b, 		posts as p where b.UserId = p.OwnerUserId  AND p.PostTypeId=2;
