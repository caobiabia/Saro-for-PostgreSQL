select  count(*) from comments as c,  		posts as p,  		postLinks as pl,          postHistory as ph,          votes as v  where p.Id = c.PostId 	and p.Id = pl.PostId     and p.Id = ph.PostId     and p.Id = v.PostId  AND c.CreationDate>='2010-08-12 01:55:20'::timestamp  AND c.CreationDate<='2014-09-13 21:49:10'::timestamp  AND ph.CreationDate>='2010-08-05 10:42:44'::timestamp  AND ph.CreationDate<='2014-09-11 11:41:08'::timestamp  AND p.Score<=37  AND p.CommentCount=3  AND v.CreationDate>='2010-07-20 00:00:00'::timestamp  AND v.CreationDate<='2014-09-11 00:00:00'::timestamp;