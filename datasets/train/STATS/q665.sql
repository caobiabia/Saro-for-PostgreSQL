select  count(*) from comments as c,  		posts as p,  		postLinks as pl,          postHistory as ph,          votes as v  where p.Id = c.PostId 	and p.Id = pl.PostId     and p.Id = ph.PostId     and p.Id = v.PostId  AND c.Score=0  AND pl.LinkTypeId=1  AND pl.CreationDate>='2011-07-14 03:52:51'::timestamp  AND p.FavoriteCount>=0  AND v.CreationDate>='2010-07-20 00:00:00'::timestamp;