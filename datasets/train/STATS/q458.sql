select  count(*) from comments as c,  		posts as p,  		postLinks as pl where  c.UserId = p.OwnerUserId 	and p.Id = pl.PostId  AND c.CreationDate>='2010-07-24 12:42:31'::timestamp  AND c.CreationDate<='2014-09-12 10:01:05'::timestamp  AND pl.LinkTypeId=1  AND p.Score>=-2  AND p.Score<=22  AND p.CommentCount>=0  AND p.CommentCount<=11  AND p.CreationDate<='2014-09-11 08:44:59'::timestamp;