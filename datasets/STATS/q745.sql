select  count(*) from comments as c,  		posts as p,          postLinks as pl,          postHistory as ph,          votes as v,          badges as b  where p.Id = c.PostId     and p.Id = pl.RelatedPostId     and p.Id = ph.PostId     and p.Id = v.PostId 	and b.UserId = c.UserId  AND c.CreationDate>='2010-08-11 00:37:42'::timestamp  AND ph.PostHistoryTypeId=2  AND ph.CreationDate>='2010-08-11 14:06:20'::timestamp  AND pl.LinkTypeId=1  AND pl.CreationDate>='2011-04-14 05:58:16'::timestamp  AND pl.CreationDate<='2014-07-24 00:05:16'::timestamp  AND p.Score<=21  AND p.CommentCount<=11  AND p.FavoriteCount>=0  AND v.VoteTypeId=2;