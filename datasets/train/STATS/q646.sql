select  count(*) from badges as b, 		posts as p where b.UserId = p.OwnerUserId  AND p.PostTypeId=2  AND p.CreationDate<='2014-08-27 17:13:50'::timestamp;
