select  count(*) from comments as c,  		posts as p,  		postLinks as pl where  c.UserId = p.OwnerUserId 	and p.Id = pl.PostId  AND p.PostTypeId=1  AND p.CreationDate<='2014-09-09 19:02:37'::timestamp;
