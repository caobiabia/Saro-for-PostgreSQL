select  count(*) from comments as c,          postLinks as pl,          posts as p,  		users as u,  		badges as b   where p.Id = pl.RelatedPostId 	and p.Id = c.PostId 	and u.Id = b.UserId 	and u.Id = p.OwnerUserId  AND b.Date>='2010-07-19 19:39:09'::timestamp  AND b.Date<='2014-09-05 05:41:20'::timestamp  AND c.Score=0  AND pl.LinkTypeId=1  AND pl.CreationDate>='2010-09-02 07:13:57'::timestamp  AND pl.CreationDate<='2014-07-31 22:22:31'::timestamp  AND p.PostTypeId=2  AND p.Score>=-3  AND p.AnswerCount>=0  AND p.AnswerCount<=3  AND p.FavoriteCount>=0  AND p.FavoriteCount<=3  AND u.Reputation<=126  AND u.CreationDate>='2010-07-29 02:21:04'::timestamp;