select  count(*) from comments as c,  		posts as p,  		postLinks as pl where  c.UserId = p.OwnerUserId 	and p.Id = pl.PostId  AND c.Score=0  AND p.CreationDate<='2014-09-12 17:29:35'::timestamp;
