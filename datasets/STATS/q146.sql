select  count(*) from comments as c,  		posts as p,  		postLinks as pl,          postHistory as ph,          votes as v  where p.Id = c.PostId 	and p.Id = pl.PostId     and p.Id = ph.PostId     and p.Id = v.PostId  AND c.CreationDate<='2014-09-11 21:21:39'::timestamp  AND ph.PostHistoryTypeId=5  AND pl.LinkTypeId=1  AND pl.CreationDate>='2011-06-07 12:11:32'::timestamp  AND p.Score>=0  AND p.AnswerCount>=0;