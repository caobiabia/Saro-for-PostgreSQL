select  count(*) from comments as c,  		posts as p,  		postLinks as pl,          postHistory as ph,          votes as v  where p.Id = c.PostId 	and p.Id = pl.PostId     and p.Id = ph.PostId     and p.Id = v.PostId  AND c.CreationDate<='2014-09-05 09:11:30'::timestamp  AND pl.CreationDate>='2010-10-01 23:13:10'::timestamp  AND p.PostTypeId=2  AND p.AnswerCount=0  AND v.VoteTypeId=2;