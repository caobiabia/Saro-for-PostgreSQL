select  count(*) from comments as c,  		posts as p,  		postLinks as pl,          postHistory as ph,          votes as v  where p.Id = c.PostId 	and p.Id = pl.PostId     and p.Id = ph.PostId     and p.Id = v.PostId  AND ph.PostHistoryTypeId=1  AND pl.LinkTypeId=1  AND p.CommentCount>=0  AND v.VoteTypeId=2  AND v.CreationDate<='2014-09-12 00:00:00'::timestamp;